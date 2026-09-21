#include "yue2/autoregressive.h"

#include "yue2/generation.h"
#include "gguf_model.h"
#include "lora.h"

#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <mutex>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

namespace yue2 {
namespace {

constexpr int kHidden = 2048;
constexpr int kIntermediate = 6144;
constexpr int kLayers = 28;
constexpr int kHeads = 16;
constexpr int kKvHeads = 8;
constexpr int kHeadDim = 128;
constexpr float kNormEpsilon = 1.0e-6F;
constexpr float kRopeTheta = 1000000.0F;
// Decode and the NAR flow step both build about 1090 nodes, and every graph
// here shares one scheduler, whose per-run reset memsets three arrays sized by
// this. At 32768 that was most of a megabyte of clearing per decode token to
// describe a graph thirty times smaller.
//
// The headroom is for adapters: a LoRA adds nodes to each of its ~196 targets,
// so roughly 600 per stacked adapter. This leaves room for several at once.
// Overflowing it aborts inside ggml rather than failing gracefully, so check
// with YUE2_DEBUG_GRAPH_NODES before trimming it further.
constexpr std::size_t kGraphSize = 8192;

std::runtime_error ar_error(const std::string & message) {
    return std::runtime_error("[yue2:ar] " + message);
}

ggml_tensor * f32(ggml_context * context, ggml_tensor * tensor) {
    return tensor->type == GGML_TYPE_F32 ? tensor : ggml_cast(context, tensor, GGML_TYPE_F32);
}

ggml_tensor * rms_norm(
    ggml_context * context,
    ggml_tensor * value,
    ggml_tensor * weight,
    float epsilon = kNormEpsilon) {
    return ggml_mul(context, ggml_rms_norm(context, value, epsilon), f32(context, weight));
}

ggml_tensor * linear(
    ggml_context * context,
    const detail::LoraStack & loras,
    ggml_tensor * value,
    ggml_tensor * weight) {
    return loras.linear(context, weight, value);
}

ggml_tensor * manual_attention(
    ggml_context * context,
    ggml_tensor * query,
    ggml_tensor * key,
    ggml_tensor * value,
    ggml_tensor * mask) {
    const float scale = 1.0F / std::sqrt(static_cast<float>(kHeadDim));
    auto * scores = ggml_mul_mat(context, key, query);
    scores = ggml_soft_max_ext(context, scores, mask, scale, 0.0F);
    auto * transposed_value = ggml_cont(context, ggml_transpose(context, value));
    auto * attended = ggml_mul_mat(context, transposed_value, scores);
    return ggml_cont(context, ggml_permute(context, attended, 0, 2, 1, 3));
}

ggml_tensor * attention(
    ggml_context * context,
    detail::GgufModel & model,
    const detail::LoraStack & loras,
    ggml_tensor * value,
    ggml_tensor * positions,
    ggml_tensor * mask,
    int layer,
    bool flash) {
    const std::string prefix = "model.layers." + std::to_string(layer) + ".self_attn.";
    auto * query = linear(context, loras, value, model.get(prefix + "q_proj.weight"));
    auto * key = linear(context, loras, value, model.get(prefix + "k_proj.weight"));
    auto * projected_value = linear(context, loras, value, model.get(prefix + "v_proj.weight"));
    const auto steps = value->ne[1];
    query = ggml_reshape_3d(context, query, kHeadDim, kHeads, steps);
    key = ggml_reshape_3d(context, key, kHeadDim, kKvHeads, steps);
    projected_value = ggml_reshape_3d(
        context, projected_value, kHeadDim, kKvHeads, steps);
    query = rms_norm(context, query, model.get(prefix + "q_norm.weight"));
    key = rms_norm(context, key, model.get(prefix + "k_norm.weight"));
    query = ggml_rope_ext(
        context, query, positions, nullptr, kHeadDim, GGML_ROPE_TYPE_NEOX, 0,
        kRopeTheta, 1.0F, 0.0F, 1.0F, 0.0F, 0.0F);
    key = ggml_rope_ext(
        context, key, positions, nullptr, kHeadDim, GGML_ROPE_TYPE_NEOX, 0,
        kRopeTheta, 1.0F, 0.0F, 1.0F, 0.0F, 0.0F);
    query = ggml_permute(context, query, 0, 2, 1, 3);
    key = ggml_permute(context, key, 0, 2, 1, 3);
    projected_value = ggml_permute(context, projected_value, 0, 2, 1, 3);

    ggml_tensor * attended = nullptr;
    if (flash) {
        key = ggml_cast(context, key, GGML_TYPE_F16);
        projected_value = ggml_cast(context, projected_value, GGML_TYPE_F16);
        attended = ggml_flash_attn_ext(
            context, query, key, projected_value, mask,
            1.0F / std::sqrt(static_cast<float>(kHeadDim)), 0.0F, 0.0F);
        ggml_flash_attn_ext_set_prec(attended, GGML_PREC_F32);
    } else {
        attended = manual_attention(context, query, key, projected_value, mask);
    }
    attended = ggml_reshape_2d(context, attended, kHidden, steps);
    return linear(context, loras, attended, model.get(prefix + "o_proj.weight"));
}

class KvCache {
public:
    KvCache(ggml_backend_t backend, std::size_t capacity)
        : capacity_(capacity) {
        const std::size_t tensor_count = 2 * kLayers;
        const ggml_init_params params = {
            tensor_count * ggml_tensor_overhead() + 1024, nullptr, true};
        context_ = ggml_init(params);
        if (!context_) throw ar_error("could not create KV-cache context");
        key_.reserve(kLayers);
        value_.reserve(kLayers);
        for (int layer = 0; layer < kLayers; ++layer) {
            auto * key = ggml_new_tensor_3d(
                context_, GGML_TYPE_F16, kHeadDim,
                static_cast<std::int64_t>(capacity), kKvHeads);
            auto * value = ggml_new_tensor_3d(
                context_, GGML_TYPE_F16, kHeadDim,
                static_cast<std::int64_t>(capacity), kKvHeads);
            ggml_set_name(key, ("yue2.ar.cache.key." + std::to_string(layer)).c_str());
            ggml_set_name(value, ("yue2.ar.cache.value." + std::to_string(layer)).c_str());
            key_.push_back(key);
            value_.push_back(value);
        }
        buffer_ = ggml_backend_alloc_ctx_tensors(context_, backend);
        if (!buffer_) {
            ggml_free(context_);
            context_ = nullptr;
            throw ar_error("could not allocate KV cache");
        }
        ggml_backend_buffer_clear(buffer_, 0);
    }

    ~KvCache() {
        if (buffer_) ggml_backend_buffer_free(buffer_);
        if (context_) ggml_free(context_);
    }
    KvCache(const KvCache &) = delete;
    KvCache & operator=(const KvCache &) = delete;

    ggml_tensor * key(int layer) const { return key_[static_cast<std::size_t>(layer)]; }
    ggml_tensor * value(int layer) const { return value_[static_cast<std::size_t>(layer)]; }
    std::size_t capacity() const noexcept { return capacity_; }
    std::size_t position() const noexcept { return position_; }
    void advance(std::size_t count) noexcept { position_ += count; }

private:
    ggml_context * context_ = nullptr;
    ggml_backend_buffer_t buffer_ = nullptr;
    std::vector<ggml_tensor *> key_;
    std::vector<ggml_tensor *> value_;
    std::size_t capacity_ = 0;
    std::size_t position_ = 0;
};

ggml_tensor * cached_attention(
    ggml_context * context,
    ggml_cgraph * graph,
    detail::GgufModel & model,
    const detail::LoraStack & loras,
    KvCache & cache,
    ggml_tensor * value,
    ggml_tensor * positions,
    ggml_tensor * cache_rows,
    ggml_tensor * mask,
    int layer,
    bool flash) {
    const std::string prefix = "model.layers." + std::to_string(layer) + ".self_attn.";
    auto * query = linear(context, loras, value, model.get(prefix + "q_proj.weight"));
    auto * key = linear(context, loras, value, model.get(prefix + "k_proj.weight"));
    auto * projected_value = linear(context, loras, value, model.get(prefix + "v_proj.weight"));
    const auto steps = value->ne[1];
    const auto total = static_cast<std::int64_t>(cache.position()) + steps;
    query = ggml_reshape_3d(context, query, kHeadDim, kHeads, steps);
    key = ggml_reshape_3d(context, key, kHeadDim, kKvHeads, steps);
    projected_value = ggml_reshape_3d(
        context, projected_value, kHeadDim, kKvHeads, steps);
    query = rms_norm(context, query, model.get(prefix + "q_norm.weight"));
    key = rms_norm(context, key, model.get(prefix + "k_norm.weight"));
    query = ggml_rope_ext(
        context, query, positions, nullptr, kHeadDim, GGML_ROPE_TYPE_NEOX, 0,
        kRopeTheta, 1.0F, 0.0F, 1.0F, 0.0F, 0.0F);
    key = ggml_rope_ext(
        context, key, positions, nullptr, kHeadDim, GGML_ROPE_TYPE_NEOX, 0,
        kRopeTheta, 1.0F, 0.0F, 1.0F, 0.0F, 0.0F);
    query = ggml_permute(context, query, 0, 2, 1, 3);
    key = ggml_cont(context, ggml_permute(context, key, 0, 2, 1, 3));
    projected_value = ggml_cont(
        context, ggml_permute(context, projected_value, 0, 2, 1, 3));
    ggml_build_forward_expand(
        graph, ggml_set_rows(context, cache.key(layer), key, cache_rows));
    ggml_build_forward_expand(
        graph, ggml_set_rows(context, cache.value(layer), projected_value, cache_rows));

    const auto * key_cache = cache.key(layer);
    const auto * value_cache = cache.value(layer);
    auto * used_key = ggml_view_3d(
        context, cache.key(layer), kHeadDim, total, kKvHeads,
        key_cache->nb[1], key_cache->nb[2], 0);
    auto * used_value = ggml_view_3d(
        context, cache.value(layer), kHeadDim, total, kKvHeads,
        value_cache->nb[1], value_cache->nb[2], 0);
    ggml_tensor * attended = nullptr;
    if (flash) {
        attended = ggml_flash_attn_ext(
            context, query, used_key, used_value, mask,
            1.0F / std::sqrt(static_cast<float>(kHeadDim)), 0.0F, 0.0F);
        ggml_flash_attn_ext_set_prec(attended, GGML_PREC_F32);
    } else {
        attended = manual_attention(context, query, used_key, used_value, mask);
    }
    attended = ggml_reshape_2d(context, attended, kHidden, steps);
    return linear(context, loras, attended, model.get(prefix + "o_proj.weight"));
}

ggml_tensor * mlp(
    ggml_context * context,
    detail::GgufModel & model,
    const detail::LoraStack & loras,
    ggml_tensor * value,
    int layer) {
    const std::string prefix = "model.layers." + std::to_string(layer) + ".mlp.";
    auto * gate = linear(context, loras, value, model.get(prefix + "gate_proj.weight"));
    auto * up = linear(context, loras, value, model.get(prefix + "up_proj.weight"));
    auto * activated = ggml_swiglu_split(context, gate, up);
    return linear(context, loras, activated, model.get(prefix + "down_proj.weight"));
}

void require_shape(
    detail::GgufModel & model,
    const std::string & name,
    std::int64_t ne0,
    std::int64_t ne1 = 1) {
    const auto * tensor = model.get(name);
    if (tensor->ne[0] != ne0 || tensor->ne[1] != ne1) {
        throw ar_error("unexpected tensor shape: " + name);
    }
}

} // namespace

class AutoregressiveState {
public:
    AutoregressiveState(const std::string & path, const AutoregressiveOptions & options)
        : model(detail::load_gguf_raw(
              path.c_str(), options.device.empty() ? nullptr : options.device.c_str(),
              options.threads)),
          loras(model, options.lora_adapters) {
        if (model.string("general.architecture") != "yue2" ||
            model.string("yue2.component") != "generation" ||
            model.u32("yue2.vocab_size") != kGenerationVocabSize ||
            model.u32("yue2.context_length") != 24576 ||
            model.u32("yue2.embedding_length") != kHidden ||
            model.u32("yue2.block_count") != kLayers ||
            model.u32("yue2.attention.head_count") != kHeads ||
            model.u32("yue2.attention.head_count_kv") != kKvHeads) {
            throw ar_error("unsupported YuE2 generation model");
        }
        require_shape(model, "model.embed_tokens.weight", kHidden, kGenerationVocabSize);
        require_shape(model, "lm_head.weight", kHidden, kGenerationVocabSize);
        for (int layer = 0; layer < kLayers; ++layer) {
            const auto prefix = "model.layers." + std::to_string(layer) + ".";
            require_shape(model, prefix + "input_layernorm.weight", kHidden);
            require_shape(model, prefix + "post_attention_layernorm.weight", kHidden);
            require_shape(model, prefix + "self_attn.q_proj.weight", kHidden, kHidden);
            require_shape(model, prefix + "self_attn.k_proj.weight", kHidden, kKvHeads * kHeadDim);
            require_shape(model, prefix + "self_attn.v_proj.weight", kHidden, kKvHeads * kHeadDim);
            require_shape(model, prefix + "self_attn.o_proj.weight", kHidden, kHidden);
            require_shape(model, prefix + "self_attn.q_norm.weight", kHeadDim);
            require_shape(model, prefix + "self_attn.k_norm.weight", kHeadDim);
            require_shape(model, prefix + "mlp.gate_proj.weight", kHidden, kIntermediate);
            require_shape(model, prefix + "mlp.up_proj.weight", kHidden, kIntermediate);
            require_shape(model, prefix + "mlp.down_proj.weight", kIntermediate, kHidden);
            require_shape(model, prefix + "nar_input_layernorm.weight", kHidden);
            require_shape(model, prefix + "nar_pre_mlp_layernorm.weight", kHidden);
            require_shape(model, prefix + "nar_self_attn.q_proj.weight", kHidden, kHidden);
            require_shape(
                model, prefix + "nar_self_attn.k_proj.weight",
                kHidden, kKvHeads * kHeadDim);
            require_shape(
                model, prefix + "nar_self_attn.v_proj.weight",
                kHidden, kKvHeads * kHeadDim);
            require_shape(model, prefix + "nar_self_attn.o_proj.weight", kHidden, kHidden);
            require_shape(model, prefix + "nar_self_attn.q_norm.weight", kHeadDim);
            require_shape(model, prefix + "nar_self_attn.k_norm.weight", kHeadDim);
            require_shape(model, prefix + "nar_mlp.gate_proj.weight", kHidden, kIntermediate);
            require_shape(model, prefix + "nar_mlp.up_proj.weight", kHidden, kIntermediate);
            require_shape(model, prefix + "nar_mlp.down_proj.weight", kIntermediate, kHidden);
        }
        require_shape(model, "model.norm.weight", kHidden);
        require_shape(model, "vae2llm.weight", 64, kHidden);
        require_shape(model, "vae2llm.bias", kHidden);
        require_shape(model, "time_embedder.mlp.0.weight", 256, kHidden);
        require_shape(model, "time_embedder.mlp.0.bias", kHidden);
        require_shape(model, "time_embedder.mlp.2.weight", kHidden, kHidden);
        require_shape(model, "time_embedder.mlp.2.bias", kHidden);
        require_shape(model, "latent_pos_embed.pe", kHidden, 24576);
        require_shape(model, "llm2vae.weight", kHidden, 64);
        require_shape(model, "llm2vae.bias", 64);

        ggml_backend_t backends[] = {model.backend(), nullptr};
        std::size_t backend_count = 1;
        auto * device = ggml_backend_get_device(model.backend());
        const bool accelerator =
            device && ggml_backend_dev_type(device) != GGML_BACKEND_DEVICE_TYPE_CPU;
        flash = accelerator && options.flash_attention;
        if (accelerator) {
            cpu_backend = detail::make_backend("cpu", options.threads, false);
            backends[backend_count++] = cpu_backend;
        }
        scheduler = ggml_backend_sched_new(
            backends, nullptr, backend_count, kGraphSize, false, true);
        if (!scheduler) throw ar_error("could not create GGML scheduler");
    }

    ~AutoregressiveState() {
        if (scheduler) ggml_backend_sched_free(scheduler);
        if (cpu_backend) ggml_backend_free(cpu_backend);
    }

    std::vector<float> run(const std::vector<std::int32_t> & ids) {
        std::lock_guard<std::mutex> lock(mutex);
        if (ids.empty() || ids.size() > 24576) {
            throw std::invalid_argument("YuE2 AR token count must be in [1,24576]");
        }
        for (const auto id : ids) {
            if (id < 0 || id >= kGenerationVocabSize) {
                throw std::invalid_argument("YuE2 AR token ID is out of range");
            }
        }

        auto & storage = graph_storage();
        const ggml_init_params params = {storage.size(), storage.data(), true};
        std::unique_ptr<ggml_context, decltype(&ggml_free)> context(
            ggml_init(params), ggml_free);
        if (!context) throw ar_error("could not create graph context");
        auto * graph = ggml_new_graph_custom(context.get(), kGraphSize, false);
        const auto steps = static_cast<std::int64_t>(ids.size());
        auto * token_ids = ggml_new_tensor_1d(context.get(), GGML_TYPE_I32, steps);
        auto * positions = ggml_new_tensor_1d(context.get(), GGML_TYPE_I32, steps);
        auto * mask = ggml_new_tensor_2d(
            context.get(), flash ? GGML_TYPE_F16 : GGML_TYPE_F32, steps, steps);
        for (auto * input : {token_ids, positions, mask}) ggml_set_input(input);
        ggml_set_name(token_ids, "yue2.ar.token_ids");
        ggml_set_name(positions, "yue2.ar.positions");
        ggml_set_name(mask, "yue2.ar.causal_mask");

        auto * hidden = ggml_get_rows(
            context.get(), model.get("model.embed_tokens.weight"), token_ids);
        for (int layer = 0; layer < kLayers; ++layer) {
            const auto prefix = "model.layers." + std::to_string(layer) + ".";
            auto * normalized = rms_norm(
                context.get(), hidden, model.get(prefix + "input_layernorm.weight"));
            hidden = ggml_add(
                context.get(), hidden,
                attention(
                    context.get(), model, loras, normalized, positions, mask,
                    layer, flash));
            normalized = rms_norm(
                context.get(), hidden,
                model.get(prefix + "post_attention_layernorm.weight"));
            hidden = ggml_add(
                context.get(), hidden,
                mlp(context.get(), model, loras, normalized, layer));
        }
        hidden = rms_norm(context.get(), hidden, model.get("model.norm.weight"));
        auto * last_hidden = ggml_view_1d(
            context.get(), hidden, kHidden,
            static_cast<std::size_t>(steps - 1) * hidden->nb[1]);
        auto * output = linear(
            context.get(), loras, last_hidden, model.get("lm_head.weight"));
        ggml_set_name(output, "yue2.ar.logits");
        ggml_build_forward_expand(graph, output);

        ggml_backend_sched_reset(scheduler);
        if (!ggml_backend_sched_alloc_graph(scheduler, graph)) {
            throw ar_error("could not allocate forward graph");
        }
        std::vector<std::int32_t> position_data(ids.size());
        for (std::size_t index = 0; index < ids.size(); ++index) {
            position_data[index] = static_cast<std::int32_t>(index);
        }
        ggml_backend_tensor_set(token_ids, ids.data(), 0, ids.size() * sizeof(ids.front()));
        ggml_backend_tensor_set(
            positions, position_data.data(), 0,
            position_data.size() * sizeof(position_data.front()));
        if (flash) {
            std::vector<ggml_fp16_t> mask_data(ids.size() * ids.size());
            for (std::int64_t query = 0; query < steps; ++query) {
                for (std::int64_t key = 0; key < steps; ++key) {
                    mask_data[static_cast<std::size_t>(query * steps + key)] =
                        ggml_fp32_to_fp16(key <= query ? 0.0F :
                            -std::numeric_limits<float>::infinity());
                }
            }
            ggml_backend_tensor_set(
                mask, mask_data.data(), 0, mask_data.size() * sizeof(mask_data.front()));
        } else {
            std::vector<float> mask_data(ids.size() * ids.size());
            for (std::int64_t query = 0; query < steps; ++query) {
                for (std::int64_t key = 0; key < steps; ++key) {
                    mask_data[static_cast<std::size_t>(query * steps + key)] =
                        key <= query ? 0.0F : -std::numeric_limits<float>::infinity();
                }
            }
            ggml_backend_tensor_set(
                mask, mask_data.data(), 0, mask_data.size() * sizeof(mask_data.front()));
        }
        const auto status = ggml_backend_sched_graph_compute(scheduler, graph);
        if (status != GGML_STATUS_SUCCESS) {
            throw ar_error(std::string("forward graph failed: ") + ggml_status_to_string(status));
        }
        std::vector<float> logits(kGenerationVocabSize);
        ggml_backend_tensor_get(output, logits.data(), 0, logits.size() * sizeof(float));
        return logits;
    }

    std::vector<float> append(KvCache & cache, const std::vector<std::int32_t> & ids) {
        std::lock_guard<std::mutex> lock(mutex);
        return append_locked(cache, ids);
    }

    std::vector<float> append_locked(
        KvCache & cache,
        const std::vector<std::int32_t> & ids) {
        if (ids.empty() || cache.position() + ids.size() > cache.capacity()) {
            throw std::invalid_argument("YuE2 AR append exceeds the KV-cache capacity");
        }
        for (const auto id : ids) {
            if (id < 0 || id >= kGenerationVocabSize) {
                throw std::invalid_argument("YuE2 AR token ID is out of range");
            }
        }

        auto & storage = graph_storage();
        const ggml_init_params params = {storage.size(), storage.data(), true};
        std::unique_ptr<ggml_context, decltype(&ggml_free)> context(
            ggml_init(params), ggml_free);
        if (!context) throw ar_error("could not create cached graph context");
        auto * graph = ggml_new_graph_custom(context.get(), kGraphSize, false);
        const auto steps = static_cast<std::int64_t>(ids.size());
        const auto cached = static_cast<std::int64_t>(cache.position());
        const auto total = cached + steps;
        auto * token_ids = ggml_new_tensor_1d(context.get(), GGML_TYPE_I32, steps);
        auto * positions = ggml_new_tensor_1d(context.get(), GGML_TYPE_I32, steps);
        auto * cache_rows = ggml_new_tensor_1d(context.get(), GGML_TYPE_I64, steps);
        auto * mask = ggml_new_tensor_2d(
            context.get(), flash ? GGML_TYPE_F16 : GGML_TYPE_F32, total, steps);
        for (auto * input : {token_ids, positions, cache_rows, mask}) ggml_set_input(input);
        ggml_set_name(token_ids, "yue2.ar.cached_token_ids");
        ggml_set_name(positions, "yue2.ar.cached_positions");
        ggml_set_name(cache_rows, "yue2.ar.cache_rows");
        ggml_set_name(mask, "yue2.ar.cached_causal_mask");

        auto * hidden = ggml_get_rows(
            context.get(), model.get("model.embed_tokens.weight"), token_ids);
        for (int layer = 0; layer < kLayers; ++layer) {
            const auto prefix = "model.layers." + std::to_string(layer) + ".";
            auto * normalized = rms_norm(
                context.get(), hidden, model.get(prefix + "input_layernorm.weight"));
            hidden = ggml_add(
                context.get(), hidden,
                cached_attention(
                    context.get(), graph, model, loras, cache, normalized, positions,
                    cache_rows, mask, layer, flash));
            normalized = rms_norm(
                context.get(), hidden,
                model.get(prefix + "post_attention_layernorm.weight"));
            hidden = ggml_add(
                context.get(), hidden,
                mlp(context.get(), model, loras, normalized, layer));
        }
        hidden = rms_norm(context.get(), hidden, model.get("model.norm.weight"));
        auto * last_hidden = ggml_view_1d(
            context.get(), hidden, kHidden,
            static_cast<std::size_t>(steps - 1) * hidden->nb[1]);
        auto * output = linear(
            context.get(), loras, last_hidden, model.get("lm_head.weight"));
        ggml_set_name(output, "yue2.ar.cached_logits");
        ggml_build_forward_expand(graph, output);
        static const bool report_nodes = std::getenv("YUE2_DEBUG_GRAPH_NODES") != nullptr;
        if (report_nodes) {
            static bool reported = false;
            if (!reported) {
                reported = true;
                std::fprintf(stderr, "[yue2] decode graph nodes: %d of %zu capacity\n",
                             ggml_graph_n_nodes(graph), kGraphSize);
            }
        }

        ggml_backend_sched_reset(scheduler);
        if (!ggml_backend_sched_alloc_graph(scheduler, graph)) {
            throw ar_error("could not allocate cached forward graph");
        }
        std::vector<std::int32_t> position_data(ids.size());
        std::vector<std::int64_t> row_data(ids.size());
        for (std::size_t index = 0; index < ids.size(); ++index) {
            position_data[index] = static_cast<std::int32_t>(cached + index);
            row_data[index] = cached + static_cast<std::int64_t>(index);
        }
        ggml_backend_tensor_set(token_ids, ids.data(), 0, ids.size() * sizeof(ids.front()));
        ggml_backend_tensor_set(
            positions, position_data.data(), 0,
            position_data.size() * sizeof(position_data.front()));
        ggml_backend_tensor_set(
            cache_rows, row_data.data(), 0, row_data.size() * sizeof(row_data.front()));
        if (flash) {
            std::vector<ggml_fp16_t> mask_data(static_cast<std::size_t>(total * steps));
            for (std::int64_t query = 0; query < steps; ++query) {
                for (std::int64_t key = 0; key < total; ++key) {
                    mask_data[static_cast<std::size_t>(query * total + key)] =
                        ggml_fp32_to_fp16(key <= cached + query ? 0.0F :
                            -std::numeric_limits<float>::infinity());
                }
            }
            ggml_backend_tensor_set(
                mask, mask_data.data(), 0, mask_data.size() * sizeof(mask_data.front()));
        } else {
            std::vector<float> mask_data(static_cast<std::size_t>(total * steps));
            for (std::int64_t query = 0; query < steps; ++query) {
                for (std::int64_t key = 0; key < total; ++key) {
                    mask_data[static_cast<std::size_t>(query * total + key)] =
                        key <= cached + query ? 0.0F : -std::numeric_limits<float>::infinity();
                }
            }
            ggml_backend_tensor_set(
                mask, mask_data.data(), 0, mask_data.size() * sizeof(mask_data.front()));
        }
        const auto status = ggml_backend_sched_graph_compute(scheduler, graph);
        if (status != GGML_STATUS_SUCCESS) {
            throw ar_error(std::string("cached forward graph failed: ") +
                ggml_status_to_string(status));
        }
        std::vector<float> logits(kGenerationVocabSize);
        ggml_backend_tensor_get(output, logits.data(), 0, logits.size() * sizeof(float));
        cache.advance(ids.size());
        return logits;
    }



    std::vector<float> solve_flow_chunk(
        const std::vector<std::int32_t> & ar_tokens,
        const std::vector<float> & noise,
        std::uint32_t ode_steps,
        const AutoregressiveControl & control);

    // Scratch for the per-call ggml context. Building a graph needs room for
    // every tensor struct it might create, which at kGraphSize is about 13MB.
    // Decode builds one of these per token, so allocating and zero-filling it
    // each time cost more than some of the work it was describing. Every
    // builder holds the mutex, so one buffer serves them all; ggml bump
    // allocates from it and does not require it to be cleared.
    std::vector<std::uint8_t> & graph_storage() {
        const std::size_t needed = ggml_tensor_overhead() * kGraphSize +
            ggml_graph_overhead_custom(kGraphSize, false);
        if (graph_storage_.size() < needed) graph_storage_.resize(needed);
        return graph_storage_;
    }

    std::vector<std::uint8_t> graph_storage_;
    detail::GgufModel model;
    detail::LoraStack loras;
    ggml_backend_t cpu_backend = nullptr;
    ggml_backend_sched_t scheduler = nullptr;
    bool flash = false;
    std::mutex mutex;
};

namespace {

ggml_tensor * linear_bias(
    ggml_context * context,
    const detail::LoraStack & loras,
    ggml_tensor * value,
    ggml_tensor * weight,
    ggml_tensor * bias) {
    return ggml_add(
        context, linear(context, loras, value, weight),
        f32(context, loras.bias(context, bias)));
}

ggml_tensor * nar_mlp(
    ggml_context * context,
    detail::GgufModel & model,
    const detail::LoraStack & loras,
    ggml_tensor * value,
    int layer) {
    const std::string prefix = "model.layers." + std::to_string(layer) + ".nar_mlp.";
    auto * gate = linear(context, loras, value, model.get(prefix + "gate_proj.weight"));
    auto * up = linear(context, loras, value, model.get(prefix + "up_proj.weight"));
    return linear(
        context, loras, ggml_swiglu_split(context, gate, up),
        model.get(prefix + "down_proj.weight"));
}

ggml_tensor * nar_attention(
    ggml_context * context,
    AutoregressiveState & state,
    KvCache & ar_cache,
    ggml_tensor * value,
    ggml_tensor * positions,
    int layer) {
    const std::string prefix =
        "model.layers." + std::to_string(layer) + ".nar_self_attn.";
    auto * query = linear(
        context, state.loras, value, state.model.get(prefix + "q_proj.weight"));
    auto * key = linear(
        context, state.loras, value, state.model.get(prefix + "k_proj.weight"));
    auto * projected_value = linear(
        context, state.loras, value, state.model.get(prefix + "v_proj.weight"));
    const auto steps = value->ne[1];
    query = ggml_reshape_3d(context, query, kHeadDim, kHeads, steps);
    key = ggml_reshape_3d(context, key, kHeadDim, kKvHeads, steps);
    projected_value = ggml_reshape_3d(
        context, projected_value, kHeadDim, kKvHeads, steps);
    query = rms_norm(context, query, state.model.get(prefix + "q_norm.weight"));
    key = rms_norm(context, key, state.model.get(prefix + "k_norm.weight"));
    query = ggml_rope_ext(
        context, query, positions, nullptr, kHeadDim, GGML_ROPE_TYPE_NEOX, 0,
        kRopeTheta, 1.0F, 0.0F, 1.0F, 0.0F, 0.0F);
    key = ggml_rope_ext(
        context, key, positions, nullptr, kHeadDim, GGML_ROPE_TYPE_NEOX, 0,
        kRopeTheta, 1.0F, 0.0F, 1.0F, 0.0F, 0.0F);
    query = ggml_permute(context, query, 0, 2, 1, 3);
    key = ggml_cast(
        context, ggml_cont(context, ggml_permute(context, key, 0, 2, 1, 3)),
        GGML_TYPE_F16);
    projected_value = ggml_cast(
        context,
        ggml_cont(context, ggml_permute(context, projected_value, 0, 2, 1, 3)),
        GGML_TYPE_F16);
    key = ggml_concat(context, ar_cache.key(layer), key, 1);
    projected_value = ggml_concat(
        context, ar_cache.value(layer), projected_value, 1);
    ggml_tensor * attended = nullptr;
    if (state.flash) {
        attended = ggml_flash_attn_ext(
            context, query, key, projected_value, nullptr,
            1.0F / std::sqrt(static_cast<float>(kHeadDim)), 0.0F, 0.0F);
        ggml_flash_attn_ext_set_prec(attended, GGML_PREC_F32);
    } else {
        attended = manual_attention(context, query, key, projected_value, nullptr);
    }
    attended = ggml_reshape_2d(context, attended, kHidden, steps);
    return linear(
        context, state.loras, attended, state.model.get(prefix + "o_proj.weight"));
}

class FlowGraph {
public:
    FlowGraph(AutoregressiveState & state, KvCache & ar_cache, std::size_t frames)
        : state_(state), ar_cache_(ar_cache), frames_(frames),
          nar_length_(static_cast<std::int64_t>(frames + 2)) {
        const std::size_t context_bytes = ggml_tensor_overhead() * kGraphSize +
            ggml_graph_overhead_custom(kGraphSize, false);
        storage_.resize(context_bytes);
        const ggml_init_params params = {context_bytes, storage_.data(), true};
        context_.reset(ggml_init(params));
        if (!context_) throw ar_error("could not create NAR graph context");
        graph_ = ggml_new_graph_custom(context_.get(), kGraphSize, false);
        latent_state_ = ggml_new_tensor_2d(
            context_.get(), GGML_TYPE_F32, 64, nar_length_);
        time_features_ = ggml_new_tensor_1d(context_.get(), GGML_TYPE_F32, 256);
        positions_ = ggml_new_tensor_1d(context_.get(), GGML_TYPE_I32, nar_length_);
        for (auto * input : {latent_state_, time_features_, positions_}) {
            ggml_set_input(input);
        }
        ggml_set_name(latent_state_, "yue2.nar.latent_state");
        ggml_set_name(time_features_, "yue2.nar.time_features");
        ggml_set_name(positions_, "yue2.nar.positions");

        auto * hidden = linear_bias(
            context_.get(), state_.loras, latent_state_, state_.model.get("vae2llm.weight"),
            state_.model.get("vae2llm.bias"));
        auto * time_hidden = linear_bias(
            context_.get(), state_.loras, time_features_,
            state_.model.get("time_embedder.mlp.0.weight"),
            state_.model.get("time_embedder.mlp.0.bias"));
        time_hidden = ggml_silu(context_.get(), time_hidden);
        time_hidden = linear_bias(
            context_.get(), state_.loras, time_hidden,
            state_.model.get("time_embedder.mlp.2.weight"),
            state_.model.get("time_embedder.mlp.2.bias"));
        auto * position_hidden = ggml_view_2d(
            context_.get(), state_.model.get("latent_pos_embed.pe"), kHidden,
            nar_length_, state_.model.get("latent_pos_embed.pe")->nb[1], 0);
        hidden = ggml_add(context_.get(), hidden, time_hidden);
        hidden = ggml_add(context_.get(), hidden, f32(context_.get(), position_hidden));

        for (int layer = 0; layer < kLayers; ++layer) {
            const auto prefix = "model.layers." + std::to_string(layer) + ".";
            auto * normalized = rms_norm(
                context_.get(), hidden,
                state_.model.get(prefix + "nar_input_layernorm.weight"));
            hidden = ggml_add(
                context_.get(), hidden,
                nar_attention(
                    context_.get(), state_, ar_cache_, normalized, positions_, layer));
            normalized = rms_norm(
                context_.get(), hidden,
                state_.model.get(prefix + "nar_pre_mlp_layernorm.weight"));
            hidden = ggml_add(
                context_.get(), hidden,
                nar_mlp(
                    context_.get(), state_.model, state_.loras, normalized, layer));
        }
        hidden = rms_norm(context_.get(), hidden, state_.model.get("model.norm.weight"));
        auto * latent = linear_bias(
            context_.get(), state_.loras, hidden, state_.model.get("llm2vae.weight"),
            state_.model.get("llm2vae.bias"));
        output_ = ggml_cont(
            context_.get(),
            ggml_view_2d(
                context_.get(), latent, 64, static_cast<std::int64_t>(frames_),
                latent->nb[1], latent->nb[1]));
        ggml_set_name(output_, "yue2.nar.velocity");
        ggml_build_forward_expand(graph_, output_);

        static const bool report_flow_nodes =
            std::getenv("YUE2_DEBUG_GRAPH_NODES") != nullptr;
        if (report_flow_nodes) {
            static bool reported = false;
            if (!reported) {
                reported = true;
                std::fprintf(stderr, "[yue2] flow graph nodes: %d of %zu capacity\n",
                             ggml_graph_n_nodes(graph_), kGraphSize);
            }
        }
        ggml_backend_sched_reset(state_.scheduler);
        if (!ggml_backend_sched_alloc_graph(state_.scheduler, graph_)) {
            throw ar_error("could not allocate NAR graph");
        }
        std::vector<std::int32_t> positions(static_cast<std::size_t>(nar_length_));
        for (std::size_t index = 0; index < positions.size(); ++index) {
            positions[index] = static_cast<std::int32_t>(ar_cache_.position() + index);
        }
        ggml_backend_tensor_set(
            positions_, positions.data(), 0, positions.size() * sizeof(positions.front()));
    }

    std::vector<float> velocity(
        const std::vector<float> & state,
        float raw_t) {
        if (state.size() != frames_ * 64) {
            throw std::invalid_argument("YuE2 NAR state shape changed during solve");
        }
        std::vector<float> padded((frames_ + 2) * 64, 0.0F);
        std::copy(state.begin(), state.end(), padded.begin() + 64);
        const float shifted = 1.0F / (1.0F + std::exp(-raw_t));
        std::vector<float> features(256);
        for (std::size_t index = 0; index < 128; ++index) {
            const auto frequency = std::exp(
                -std::log(10000.0F) * static_cast<float>(index) / 128.0F);
            const auto angle = shifted * frequency;
            features[index] = std::cos(angle);
            features[index + 128] = std::sin(angle);
        }
        ggml_backend_tensor_set(
            latent_state_, padded.data(), 0, padded.size() * sizeof(padded.front()));
        ggml_backend_tensor_set(
            time_features_, features.data(), 0, features.size() * sizeof(features.front()));
        const auto status = ggml_backend_sched_graph_compute(state_.scheduler, graph_);
        if (status != GGML_STATUS_SUCCESS) {
            throw ar_error(std::string("NAR graph failed: ") + ggml_status_to_string(status));
        }
        std::vector<float> output(frames_ * 64);
        ggml_backend_tensor_get(
            output_, output.data(), 0, output.size() * sizeof(output.front()));
        return output;
    }

private:
    AutoregressiveState & state_;
    KvCache & ar_cache_;
    std::size_t frames_ = 0;
    std::int64_t nar_length_ = 0;
    std::vector<std::uint8_t> storage_;
    std::unique_ptr<ggml_context, decltype(&ggml_free)> context_{nullptr, ggml_free};
    ggml_cgraph * graph_ = nullptr;
    ggml_tensor * latent_state_ = nullptr;
    ggml_tensor * time_features_ = nullptr;
    ggml_tensor * positions_ = nullptr;
    ggml_tensor * output_ = nullptr;
};

double logit_clamped(double probability) {
    if (probability <= 0.0) return -20.0;
    if (probability >= 1.0) return 20.0;
    return std::max(-20.0, std::min(20.0, std::log(probability / (1.0 - probability))));
}

} // namespace

std::vector<float> AutoregressiveState::solve_flow_chunk(
    const std::vector<std::int32_t> & ar_tokens,
    const std::vector<float> & noise,
    std::uint32_t ode_steps,
    const AutoregressiveControl & control) {
    std::lock_guard<std::mutex> lock(mutex);
    if (ar_tokens.empty() || noise.empty() || noise.size() % 64 != 0 || ode_steps < 1) {
        throw std::invalid_argument("invalid YuE2 NAR chunk inputs");
    }
    const auto frames = noise.size() / 64;
    if (ar_tokens.size() + frames + 2 > 24576) {
        throw std::invalid_argument("YuE2 NAR chunk exceeds model context");
    }
    KvCache ar_cache(model.backend(), ar_tokens.size());
    (void)append_locked(ar_cache, ar_tokens);
    FlowGraph graph(*this, ar_cache, frames);
    auto state = noise;
    const double dt = 1.0 / static_cast<double>(ode_steps);
    for (std::uint32_t step = 0; step < ode_steps; ++step) {
        if (control.should_cancel && control.should_cancel()) {
            throw std::runtime_error("YuE2 generation cancelled");
        }
        const double t = 1.0 - static_cast<double>(step) * dt;
        const auto first = graph.velocity(state, static_cast<float>(logit_clamped(t)));
        std::vector<float> midpoint(state.size());
        for (std::size_t index = 0; index < state.size(); ++index) {
            midpoint[index] = state[index] - first[index] * static_cast<float>(dt / 2.0);
        }
        const auto second = graph.velocity(
            midpoint, static_cast<float>(logit_clamped(t - dt / 2.0)));
        for (std::size_t index = 0; index < state.size(); ++index) {
            state[index] -= second[index] * static_cast<float>(dt);
        }
        if (control.on_progress) control.on_progress(step + 1, ode_steps);
    }
    return state;
}

class AutoregressiveSession::Impl {
public:
    Impl(std::shared_ptr<AutoregressiveState> state, std::size_t capacity)
        : state(std::move(state)), cache(this->state->model.backend(), capacity) {}

    std::shared_ptr<AutoregressiveState> state;
    KvCache cache;
};

namespace {

void validate_sampling(const GenerationSampling & sampling) {
    if (!std::isfinite(sampling.temperature) || sampling.temperature < 0.0F ||
        sampling.temperature > 5.0F || !std::isfinite(sampling.top_p) ||
        sampling.top_p <= 0.0F || sampling.top_p > 1.0F || sampling.top_k < 1 ||
        !std::isfinite(sampling.repetition_penalty) ||
        sampling.repetition_penalty <= 0.0F || sampling.penalty_window < 1 ||
        sampling.penalty_window > 100 || sampling.min_tokens > sampling.max_tokens ||
        sampling.max_tokens < 1) {
        throw std::invalid_argument("invalid YuE2 AR sampling configuration");
    }
}

struct Candidate {
    std::int32_t token = 0;
    float score = 0.0F;
    double weight = 0.0;
};

std::int32_t sample_token(
    const std::vector<float> & logits,
    const GenerationSampling & sampling,
    const std::vector<std::int32_t> & history,
    std::uint32_t step,
    AutoregressivePhase phase,
    std::mt19937_64 & random,
    bool allow_stop = true) {
    const auto stop = phase == AutoregressivePhase::abc ? kAbcEndToken : kMusicEndToken;
    const auto begin = phase == AutoregressivePhase::abc ? 0 : kCodecOffset;
    const auto end = phase == AutoregressivePhase::abc ? kEodToken : kCodecOffset + kCodecSize;
    if (logits.size() != kGenerationVocabSize) {
        throw std::invalid_argument("YuE2 sampler requires full-vocabulary logits");
    }
    std::vector<std::uint8_t> counts(logits.size(), 0);
    const auto history_begin = history.size() > sampling.penalty_window
        ? history.size() - sampling.penalty_window
        : 0;
    for (std::size_t index = history_begin; index < history.size(); ++index) {
        const auto token = history[index];
        if (token >= 0 && static_cast<std::size_t>(token) < counts.size() &&
            counts[static_cast<std::size_t>(token)] < 255) {
            ++counts[static_cast<std::size_t>(token)];
        }
    }

    std::vector<Candidate> candidates;
    candidates.reserve(static_cast<std::size_t>(end - begin + 1));
    auto add = [&](std::int32_t token) {
        float score = logits[static_cast<std::size_t>(token)];
        const auto count = counts[static_cast<std::size_t>(token)];
        if (count != 0 && sampling.repetition_penalty != 1.0F) {
            const auto factor = std::pow(sampling.repetition_penalty, count);
            score = score < 0.0F ? score * factor : score / factor;
        }
        if (std::isfinite(score)) candidates.push_back({token, score, 0.0});
    };
    for (std::int32_t token = begin; token < end; ++token) add(token);
    if (allow_stop && step >= sampling.min_tokens) add(stop);
    if (candidates.empty()) throw ar_error("sampler has no finite allowed logits");

    if (sampling.temperature == 0.0F) {
        return std::max_element(
            candidates.begin(), candidates.end(),
            [](const Candidate & first, const Candidate & second) {
                return first.score < second.score ||
                    (first.score == second.score && first.token > second.token);
            })->token;
    }
    for (auto & candidate : candidates) candidate.score /= sampling.temperature;

    if (sampling.top_k < candidates.size()) {
        std::vector<float> scores;
        scores.reserve(candidates.size());
        for (const auto & candidate : candidates) scores.push_back(candidate.score);
        const auto kth = scores.begin() + static_cast<std::ptrdiff_t>(sampling.top_k - 1);
        std::nth_element(scores.begin(), kth, scores.end(), std::greater<float>());
        const auto threshold = *kth;
        candidates.erase(
            std::remove_if(
                candidates.begin(), candidates.end(),
                [threshold](const Candidate & candidate) {
                    return candidate.score < threshold;
                }),
            candidates.end());
    }
    std::sort(
        candidates.begin(), candidates.end(),
        [](const Candidate & first, const Candidate & second) {
            return first.score > second.score ||
                (first.score == second.score && first.token < second.token);
        });
    const auto maximum = candidates.front().score;
    double total = 0.0;
    for (auto & candidate : candidates) {
        candidate.weight = std::exp(static_cast<double>(candidate.score - maximum));
        total += candidate.weight;
    }
    if (!(total > 0.0) || !std::isfinite(total)) {
        throw ar_error("sampler probability mass is invalid");
    }
    if (sampling.top_p < 1.0F) {
        double cumulative_before = 0.0;
        std::size_t keep = 0;
        for (; keep < candidates.size(); ++keep) {
            if (keep > 0 && cumulative_before > sampling.top_p) break;
            cumulative_before += candidates[keep].weight / total;
        }
        candidates.resize(std::max<std::size_t>(keep, 1));
        total = 0.0;
        for (const auto & candidate : candidates) total += candidate.weight;
    }
    std::uniform_real_distribution<double> distribution(0.0, total);
    const auto draw = distribution(random);
    double cumulative = 0.0;
    for (const auto & candidate : candidates) {
        cumulative += candidate.weight;
        if (draw < cumulative) return candidate.token;
    }
    return candidates.back().token;
}

void validate_request(
    const std::vector<std::int32_t> & prefix,
    const GenerationSampling & sampling) {
    validate_sampling(sampling);
    if (prefix.empty() || prefix.size() + sampling.max_tokens > 24576) {
        throw std::invalid_argument(
            "YuE2 AR prefix plus generation budget must fit the 24576-token context");
    }
}

} // namespace

AutoregressiveSession::AutoregressiveSession(std::unique_ptr<Impl> impl)
    : impl_(std::move(impl)) {}
AutoregressiveSession::~AutoregressiveSession() = default;
AutoregressiveSession::AutoregressiveSession(AutoregressiveSession &&) noexcept = default;
AutoregressiveSession & AutoregressiveSession::operator=(AutoregressiveSession &&) noexcept = default;

std::vector<float> AutoregressiveSession::append(
    const std::vector<std::int32_t> & token_ids) {
    return impl_->state->append(impl_->cache, token_ids);
}

std::size_t AutoregressiveSession::token_count() const noexcept {
    return impl_->cache.position();
}

std::size_t AutoregressiveSession::capacity() const noexcept {
    return impl_->cache.capacity();
}

AutoregressiveModel::AutoregressiveModel(
    const std::string & model_gguf_path,
    const AutoregressiveOptions & options)
    : state_(std::make_shared<AutoregressiveState>(model_gguf_path, options)) {}

AutoregressiveModel::~AutoregressiveModel() = default;
AutoregressiveModel::AutoregressiveModel(AutoregressiveModel &&) noexcept = default;
AutoregressiveModel & AutoregressiveModel::operator=(AutoregressiveModel &&) noexcept = default;

std::vector<float> AutoregressiveModel::logits(
    const std::vector<std::int32_t> & token_ids) {
    return state_->run(token_ids);
}

std::unique_ptr<AutoregressiveSession> AutoregressiveModel::create_session(
    std::size_t capacity) {
    if (capacity < 1 || capacity > 24576) {
        throw std::invalid_argument("YuE2 AR session capacity must be in [1,24576]");
    }
    return std::unique_ptr<AutoregressiveSession>(
        new AutoregressiveSession(
            std::make_unique<AutoregressiveSession::Impl>(state_, capacity)));
}

AutoregressiveResult AutoregressiveModel::generate(
    const std::vector<std::int32_t> & prefix,
    const GenerationSampling & sampling,
    AutoregressivePhase phase,
    std::uint64_t seed,
    const AutoregressiveControl & control) {
    validate_request(prefix, sampling);
    if (control.should_cancel && control.should_cancel()) {
        throw std::runtime_error("YuE2 generation cancelled");
    }
    auto session = create_session(prefix.size() + sampling.max_tokens);
    auto logits = session->append(prefix);
    std::mt19937_64 random(seed);
    AutoregressiveResult result;
    result.tokens.reserve(sampling.max_tokens);
    const auto stop = phase == AutoregressivePhase::abc ? kAbcEndToken : kMusicEndToken;
    for (std::uint32_t step = 0; step < sampling.max_tokens; ++step) {
        if (control.should_cancel && control.should_cancel()) {
            throw std::runtime_error("YuE2 generation cancelled");
        }
        auto token = sample_token(
            logits, sampling, result.tokens, step, phase, random);
        if (token == stop && control.allow_stop && !control.allow_stop(result.tokens)) {
            token = sample_token(
                logits, sampling, result.tokens, step, phase, random, false);
        }
        if (control.on_progress) control.on_progress(step + 1, sampling.max_tokens);
        if (token == stop) {
            result.reached_end = true;
            break;
        }
        result.tokens.push_back(token);
        if (step + 1 < sampling.max_tokens) logits = session->append({token});
    }
    return result;
}

AutoregressiveResult AutoregressiveModel::generate_cfg(
    const std::vector<std::int32_t> & positive_prefix,
    const std::vector<std::int32_t> & negative_prefix,
    const GenerationSampling & sampling,
    AutoregressivePhase phase,
    float guidance_scale,
    std::uint64_t seed,
    const AutoregressiveControl & control) {
    validate_request(positive_prefix, sampling);
    validate_request(negative_prefix, sampling);
    if (!std::isfinite(guidance_scale) || guidance_scale < 0.0F || guidance_scale > 20.0F) {
        throw std::invalid_argument("YuE2 guidance scale must be finite and in [0,20]");
    }
    if (guidance_scale == 1.0F) {
        return generate(positive_prefix, sampling, phase, seed, control);
    }
    if (control.should_cancel && control.should_cancel()) {
        throw std::runtime_error("YuE2 generation cancelled");
    }
    auto positive = create_session(positive_prefix.size() + sampling.max_tokens);
    auto negative = create_session(negative_prefix.size() + sampling.max_tokens);
    auto positive_logits = positive->append(positive_prefix);
    auto negative_logits = negative->append(negative_prefix);
    std::vector<float> guided(kGenerationVocabSize);
    std::mt19937_64 random(seed);
    AutoregressiveResult result;
    result.tokens.reserve(sampling.max_tokens);
    const auto stop = phase == AutoregressivePhase::abc ? kAbcEndToken : kMusicEndToken;
    for (std::uint32_t step = 0; step < sampling.max_tokens; ++step) {
        if (control.should_cancel && control.should_cancel()) {
            throw std::runtime_error("YuE2 generation cancelled");
        }
        for (std::size_t index = 0; index < guided.size(); ++index) {
            guided[index] = negative_logits[index] +
                guidance_scale * (positive_logits[index] - negative_logits[index]);
        }
        auto token = sample_token(
            guided, sampling, result.tokens, step, phase, random);
        if (token == stop && control.allow_stop && !control.allow_stop(result.tokens)) {
            token = sample_token(
                guided, sampling, result.tokens, step, phase, random, false);
        }
        if (control.on_progress) control.on_progress(step + 1, sampling.max_tokens);
        if (token == stop) {
            result.reached_end = true;
            break;
        }
        result.tokens.push_back(token);
        if (step + 1 < sampling.max_tokens) {
            positive_logits = positive->append({token});
            negative_logits = negative->append({token});
        }
    }
    return result;
}

std::vector<float> AutoregressiveModel::synthesize_latents(
    const std::vector<std::int32_t> & generation_prefix,
    const std::vector<std::int32_t> & semantic_codec_ids,
    const std::vector<float> & noise,
    const FlowOptions & options,
    const AutoregressiveControl & control) {
    if (generation_prefix.empty() || semantic_codec_ids.empty() ||
        noise.size() != semantic_codec_ids.size() * 64 || options.ode_steps < 1 ||
        options.context_length < 1 || options.context_length > 24576 ||
        generation_prefix.size() + 3 >= options.context_length) {
        throw std::invalid_argument("invalid YuE2 flow synthesis request");
    }
    for (const auto token : generation_prefix) {
        if (token < 0 || token >= kGenerationVocabSize) {
            throw std::invalid_argument("YuE2 generation prefix token is out of range");
        }
    }
    for (const auto token : semantic_codec_ids) {
        if (token < 0 || token >= kCodecSize) {
            throw std::invalid_argument("YuE2 semantic codec ID is out of range");
        }
    }
    for (const auto value : noise) {
        if (!std::isfinite(value)) {
            throw std::invalid_argument("YuE2 flow noise contains a non-finite value");
        }
    }
    const auto chunk_frames = static_cast<std::size_t>(
        (options.context_length - generation_prefix.size() - 3) / 2);
    if (chunk_frames < 1) {
        throw std::invalid_argument("YuE2 generation prefix leaves no acoustic context");
    }

    std::vector<float> output;
    output.reserve(noise.size());
    const auto chunk_count = static_cast<std::uint32_t>(
        (semantic_codec_ids.size() + chunk_frames - 1) / chunk_frames);
    std::uint32_t chunk_index = 0;
    for (std::size_t begin = 0; begin < semantic_codec_ids.size(); begin += chunk_frames) {
        if (control.should_cancel && control.should_cancel()) {
            throw std::runtime_error("YuE2 generation cancelled");
        }
        const auto end = std::min(begin + chunk_frames, semantic_codec_ids.size());
        std::vector<std::int32_t> ar_tokens = generation_prefix;
        ar_tokens.reserve(generation_prefix.size() + (end - begin) + 1);
        for (auto index = begin; index < end; ++index) {
            ar_tokens.push_back(kCodecOffset + semantic_codec_ids[index]);
        }
        ar_tokens.push_back(kMusicEndToken);
        const auto noise_begin = noise.begin() + static_cast<std::ptrdiff_t>(begin * 64);
        const auto noise_end = noise.begin() + static_cast<std::ptrdiff_t>(end * 64);
        const std::vector<float> chunk_noise(noise_begin, noise_end);
        AutoregressiveControl chunk_control;
        chunk_control.should_cancel = control.should_cancel;
        if (control.on_progress) {
            chunk_control.on_progress = [&, chunk_index](
                std::uint32_t current,
                std::uint32_t) {
                control.on_progress(
                    chunk_index * options.ode_steps + current,
                    chunk_count * options.ode_steps);
            };
        }
        auto chunk = state_->solve_flow_chunk(
            ar_tokens, chunk_noise, options.ode_steps, chunk_control);
        output.insert(output.end(), chunk.begin(), chunk.end());
        ++chunk_index;
    }
    return output;
}

std::vector<float> AutoregressiveModel::synthesize_latents(
    const std::vector<std::int32_t> & generation_prefix,
    const std::vector<std::int32_t> & semantic_codec_ids,
    std::uint64_t seed,
    const FlowOptions & options,
    const AutoregressiveControl & control) {
    std::mt19937 random(static_cast<std::uint32_t>(seed));
    std::normal_distribution<float> normal(0.0F, 1.0F);
    std::vector<float> noise(semantic_codec_ids.size() * 64);
    for (auto & value : noise) value = normal(random);
    return synthesize_latents(
        generation_prefix, semantic_codec_ids, noise, options, control);
}

} // namespace yue2
