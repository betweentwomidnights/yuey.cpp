#include "yue2/mert2_encoder.h"
#include "yue2/sheetsage2_tokens.h"

#include "gguf_model.h"

#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <memory>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace yue2::mert2 {
namespace {

constexpr float subsampling_norm_epsilon = 1.0e-6F;
constexpr float conformer_norm_epsilon = 1.0e-5F;
constexpr std::size_t graph_size = 8192;

ggml_tensor * named(detail::GgufModel & model, const std::string & name) {
    return model.get("mert2." + name);
}

ggml_tensor * add_bias(ggml_context * context, ggml_tensor * value, ggml_tensor * bias) {
    return bias ? ggml_add(context, value, bias) : value;
}

ggml_tensor * layer_norm(
    ggml_context * context,
    ggml_tensor * value,
    ggml_tensor * weight,
    ggml_tensor * bias,
    float epsilon = subsampling_norm_epsilon) {
    return ggml_add(context, ggml_mul(context, ggml_norm(context, value, epsilon), weight), bias);
}

ggml_tensor * conv1d_channels_first(
    ggml_context * context,
    ggml_tensor * value,
    ggml_tensor * weight,
    ggml_tensor * bias,
    int stride,
    int padding,
    bool depthwise) {
    // GGML convolution uses [time, channels], while LayerNorm and Linear are
    // most natural as [channels, time]. Keep one canonical external layout and
    // make both transitions explicit.
    auto * time_major = ggml_cont(context, ggml_transpose(context, value));
    auto * convolved = depthwise
        ? ggml_conv_1d_dw(context, weight, time_major, stride, padding, 1)
        : ggml_conv_1d(context, weight, time_major, stride, padding, 1);
    auto * channels_first = ggml_cont(context, ggml_transpose(context, convolved));
    return add_bias(context, channels_first, bias);
}

ggml_tensor * linear(
    ggml_context * context,
    ggml_tensor * value,
    ggml_tensor * weight,
    ggml_tensor * bias) {
    return add_bias(context, ggml_mul_mat(context, weight, value), bias);
}

ggml_tensor * global_response_norm(
    ggml_context * context,
    ggml_tensor * value,
    ggml_tensor * weight,
    ggml_tensor * bias,
    ggml_tensor * epsilon) {
    auto * time_major = ggml_cont(context, ggml_transpose(context, value));
    auto * squared = ggml_sqr(context, time_major);
    auto * magnitude = ggml_sqrt(context, ggml_sum_rows(context, squared));
    magnitude = ggml_reshape_1d(context, magnitude, value->ne[0]);
    auto * channel_mean = ggml_mean(context, magnitude);
    auto * normalized = ggml_div(context, magnitude, ggml_add(context, channel_mean, epsilon));

    auto * weight_f32 = ggml_reshape_1d(context, ggml_cast(context, weight, GGML_TYPE_F32), value->ne[0]);
    auto * bias_f32 = ggml_reshape_1d(context, ggml_cast(context, bias, GGML_TYPE_F32), value->ne[0]);
    auto * response = ggml_mul(context, value, normalized);
    response = ggml_mul(context, response, weight_f32);
    return ggml_add(context, ggml_add(context, response, bias_f32), value);
}

ggml_tensor * convnext_layer(
    ggml_context * context,
    detail::GgufModel & model,
    ggml_tensor * value,
    int block,
    int layer,
    ggml_tensor * epsilon) {
    const std::string prefix = "sub." + std::to_string(block) +
        ".convnext_layers." + std::to_string(layer);
    auto * residual = value;
    value = conv1d_channels_first(
        context,
        value,
        named(model, prefix + ".depthwise_block.1.weight"),
        named(model, prefix + ".depthwise_block.1.bias"),
        1,
        3,
        true);
    value = layer_norm(
        context,
        value,
        named(model, prefix + ".pointwise_block.0.weight"),
        named(model, prefix + ".pointwise_block.0.bias"));
    value = linear(
        context,
        value,
        named(model, prefix + ".pointwise_block.1.weight"),
        named(model, prefix + ".pointwise_block.1.bias"));
    value = ggml_gelu_erf(context, value);
    value = global_response_norm(
        context,
        value,
        named(model, prefix + ".pointwise_block.3.weight"),
        named(model, prefix + ".pointwise_block.3.bias"),
        epsilon);
    value = linear(
        context,
        value,
        named(model, prefix + ".pointwise_block.4.weight"),
        named(model, prefix + ".pointwise_block.4.bias"));
    return ggml_add(context, residual, value);
}

ggml_tensor * attention_f32(
    ggml_context * context,
    ggml_tensor * query,
    ggml_tensor * key,
    ggml_tensor * value,
    ggml_tensor * mask,
    float scale) {
    auto * scores = ggml_mul_mat(context, key, query);
    scores = ggml_soft_max_ext(context, scores, mask, scale, 0.0F);
    auto * value_transposed = ggml_cont(context, ggml_transpose(context, value));
    auto * attended = ggml_mul_mat(context, value_transposed, scores);
    return ggml_cont(context, ggml_permute(context, attended, 0, 2, 1, 3));
}

ggml_tensor * decoder_attention(
    ggml_context * context,
    detail::GgufModel & model,
    ggml_tensor * query_input,
    ggml_tensor * key_value_input,
    const std::string & prefix,
    ggml_tensor * mask,
    bool flash) {
    constexpr int width = 512;
    constexpr int heads = 8;
    constexpr int head_width = 64;
    auto project = [&](ggml_tensor * input, const char * name) {
        return linear(
            context,
            input,
            model.get(prefix + name + ".weight"),
            model.get(prefix + name + ".bias"));
    };
    auto * query = ggml_reshape_3d(
        context, project(query_input, "q_proj"), head_width, heads, query_input->ne[1]);
    auto * key = ggml_reshape_3d(
        context, project(key_value_input, "k_proj"), head_width, heads, key_value_input->ne[1]);
    auto * value = ggml_reshape_3d(
        context, project(key_value_input, "v_proj"), head_width, heads, key_value_input->ne[1]);
    query = ggml_permute(context, query, 0, 2, 1, 3);
    key = ggml_permute(context, key, 0, 2, 1, 3);
    value = ggml_permute(context, value, 0, 2, 1, 3);

    ggml_tensor * attended = nullptr;
    constexpr float scale = 0.125F;
    if (flash) {
        key = ggml_cast(context, key, GGML_TYPE_F16);
        value = ggml_cast(context, value, GGML_TYPE_F16);
        attended = ggml_flash_attn_ext(context, query, key, value, mask, scale, 0.0F, 0.0F);
        ggml_flash_attn_ext_set_prec(attended, GGML_PREC_F32);
    } else {
        attended = attention_f32(context, query, key, value, mask, scale);
    }
    attended = ggml_reshape_2d(context, attended, width, query_input->ne[1]);
    return linear(
        context,
        attended,
        model.get(prefix + "out_proj.weight"),
        model.get(prefix + "out_proj.bias"));
}

ggml_tensor * decoder_layer(
    ggml_context * context,
    detail::GgufModel & model,
    ggml_tensor * value,
    ggml_tensor * memory,
    ggml_tensor * causal_mask,
    int layer,
    bool flash_cross_attention) {
    const std::string prefix = "sheetsage2.decoder.layers." + std::to_string(layer) + ".";
    value = ggml_add(context, value, decoder_attention(
        context, model, value, value, prefix + "self_attn.", causal_mask, false));
    value = layer_norm(
        context,
        value,
        model.get(prefix + "self_attn_layer_norm.weight"),
        model.get(prefix + "self_attn_layer_norm.bias"),
        conformer_norm_epsilon);
    value = ggml_add(context, value, decoder_attention(
        context, model, value, memory, prefix + "encoder_attn.", nullptr, flash_cross_attention));
    value = layer_norm(
        context,
        value,
        model.get(prefix + "encoder_attn_layer_norm.weight"),
        model.get(prefix + "encoder_attn_layer_norm.bias"),
        conformer_norm_epsilon);
    auto * feed_forward = linear(
        context,
        value,
        model.get(prefix + "fc1.weight"),
        model.get(prefix + "fc1.bias"));
    feed_forward = ggml_gelu_erf(context, feed_forward);
    feed_forward = linear(
        context,
        feed_forward,
        model.get(prefix + "fc2.weight"),
        model.get(prefix + "fc2.bias"));
    value = ggml_add(context, value, feed_forward);
    return layer_norm(
        context,
        value,
        model.get(prefix + "final_layer_norm.weight"),
        model.get(prefix + "final_layer_norm.bias"),
        conformer_norm_epsilon);
}

struct DecoderCache {
    DecoderCache(ggml_backend_t backend, std::int64_t max_tokens,
                 std::int64_t memory_frames, bool flash_cross)
        : max_tokens(max_tokens), memory_frames(memory_frames), flash_cross(flash_cross) {
        const std::size_t context_bytes = ggml_tensor_overhead() * 32 + 1024;
        const ggml_init_params params = {context_bytes, nullptr, true};
        context = ggml_init(params);
        if (!context) throw std::runtime_error("[yue2:sheetsage2] could not create KV-cache context");
        const auto cross_type = flash_cross ? GGML_TYPE_F16 : GGML_TYPE_F32;
        for (int layer = 0; layer < 6; ++layer) {
            self_key[static_cast<std::size_t>(layer)] =
                ggml_new_tensor_3d(context, GGML_TYPE_F32, 64, max_tokens, 8);
            self_value[static_cast<std::size_t>(layer)] =
                ggml_new_tensor_3d(context, GGML_TYPE_F32, 64, max_tokens, 8);
            cross_key[static_cast<std::size_t>(layer)] =
                ggml_new_tensor_3d(context, cross_type, 64, memory_frames, 8);
            cross_value[static_cast<std::size_t>(layer)] =
                ggml_new_tensor_3d(context, cross_type, 64, memory_frames, 8);
        }
        buffer = ggml_backend_alloc_ctx_tensors(context, backend);
        if (!buffer) {
            ggml_free(context);
            context = nullptr;
            throw std::runtime_error("[yue2:sheetsage2] could not allocate decoder KV cache");
        }
        ggml_backend_buffer_clear(buffer, 0);
    }

    ~DecoderCache() {
        if (buffer) ggml_backend_buffer_free(buffer);
        if (context) ggml_free(context);
    }

    DecoderCache(const DecoderCache &) = delete;
    DecoderCache & operator=(const DecoderCache &) = delete;

    ggml_context * context = nullptr;
    ggml_backend_buffer_t buffer = nullptr;
    std::array<ggml_tensor *, 6> self_key{};
    std::array<ggml_tensor *, 6> self_value{};
    std::array<ggml_tensor *, 6> cross_key{};
    std::array<ggml_tensor *, 6> cross_value{};
    std::int64_t max_tokens = 0;
    std::int64_t memory_frames = 0;
    bool flash_cross = false;
};

ggml_tensor * projected_query(
    ggml_context * context,
    detail::GgufModel & model,
    ggml_tensor * input,
    const std::string & prefix) {
    auto * query = linear(
        context, input, model.get(prefix + "q_proj.weight"), model.get(prefix + "q_proj.bias"));
    query = ggml_reshape_3d(context, query, 64, 8, input->ne[1]);
    return ggml_permute(context, query, 0, 2, 1, 3);
}

ggml_tensor * cached_attention_output(
    ggml_context * context,
    detail::GgufModel & model,
    ggml_tensor * query_input,
    ggml_tensor * key,
    ggml_tensor * value,
    const std::string & prefix,
    ggml_tensor * mask,
    bool flash) {
    auto * query = projected_query(context, model, query_input, prefix);
    ggml_tensor * attended = nullptr;
    if (flash) {
        attended = ggml_flash_attn_ext(context, query, key, value, mask, 0.125F, 0.0F, 0.0F);
        ggml_flash_attn_ext_set_prec(attended, GGML_PREC_F32);
    } else {
        attended = attention_f32(context, query, key, value, mask, 0.125F);
    }
    attended = ggml_reshape_2d(context, attended, 512, query_input->ne[1]);
    return linear(
        context, attended, model.get(prefix + "out_proj.weight"), model.get(prefix + "out_proj.bias"));
}

ggml_tensor * cached_decoder_layer(
    ggml_context * context,
    ggml_cgraph * graph,
    detail::GgufModel & model,
    DecoderCache & cache,
    ggml_tensor * value,
    ggml_tensor * causal_mask,
    ggml_tensor * cache_rows,
    std::int64_t cached_tokens,
    int layer) {
    const std::string prefix = "sheetsage2.decoder.layers." + std::to_string(layer) + ".";
    const std::string self_prefix = prefix + "self_attn.";
    auto project_self = [&](const char * name) {
        auto * projected = linear(
            context, value, model.get(self_prefix + name + ".weight"),
            model.get(self_prefix + name + ".bias"));
        projected = ggml_reshape_3d(context, projected, 64, 8, value->ne[1]);
        return ggml_cont(context, ggml_permute(context, projected, 0, 2, 1, 3));
    };
    auto * key = project_self("k_proj");
    auto * stored_value = project_self("v_proj");
    ggml_build_forward_expand(graph, ggml_set_rows(
        context, cache.self_key[static_cast<std::size_t>(layer)], key, cache_rows));
    ggml_build_forward_expand(graph, ggml_set_rows(
        context, cache.self_value[static_cast<std::size_t>(layer)], stored_value, cache_rows));

    const auto total_tokens = cached_tokens + value->ne[1];
    const std::size_t row_bytes = 64 * ggml_type_size(GGML_TYPE_F32);
    const std::size_t head_stride = row_bytes * static_cast<std::size_t>(cache.max_tokens);
    auto * all_key = ggml_view_3d(
        context, cache.self_key[static_cast<std::size_t>(layer)], 64, total_tokens, 8,
        row_bytes, head_stride, 0);
    auto * all_value = ggml_view_3d(
        context, cache.self_value[static_cast<std::size_t>(layer)], 64, total_tokens, 8,
        row_bytes, head_stride, 0);
    value = ggml_add(context, value, cached_attention_output(
        context, model, value, all_key, all_value, self_prefix, causal_mask, false));
    value = layer_norm(
        context, value, model.get(prefix + "self_attn_layer_norm.weight"),
        model.get(prefix + "self_attn_layer_norm.bias"), conformer_norm_epsilon);

    const std::string cross_prefix = prefix + "encoder_attn.";
    value = ggml_add(context, value, cached_attention_output(
        context, model, value, cache.cross_key[static_cast<std::size_t>(layer)],
        cache.cross_value[static_cast<std::size_t>(layer)], cross_prefix, nullptr,
        cache.flash_cross));
    value = layer_norm(
        context, value, model.get(prefix + "encoder_attn_layer_norm.weight"),
        model.get(prefix + "encoder_attn_layer_norm.bias"), conformer_norm_epsilon);
    auto * feed_forward = linear(
        context, value, model.get(prefix + "fc1.weight"), model.get(prefix + "fc1.bias"));
    feed_forward = ggml_gelu_erf(context, feed_forward);
    feed_forward = linear(
        context, feed_forward, model.get(prefix + "fc2.weight"), model.get(prefix + "fc2.bias"));
    value = ggml_add(context, value, feed_forward);
    return layer_norm(
        context, value, model.get(prefix + "final_layer_norm.weight"),
        model.get(prefix + "final_layer_norm.bias"), conformer_norm_epsilon);
}

ggml_tensor * apply_half_split_rope(
    ggml_context * context,
    ggml_tensor * value,
    ggml_tensor * cosine,
    ggml_tensor * sine) {
    const std::int64_t half = value->ne[0] / 2;
    auto * first = ggml_cont(context, ggml_view_3d(
        context, value, half, value->ne[1], value->ne[2], value->nb[1], value->nb[2], 0));
    auto * second = ggml_cont(context, ggml_view_3d(
        context, value, half, value->ne[1], value->ne[2], value->nb[1], value->nb[2],
        static_cast<std::size_t>(half) * value->nb[0]));
    auto * rotated_first = ggml_sub(
        context, ggml_mul(context, first, cosine), ggml_mul(context, second, sine));
    auto * rotated_second = ggml_add(
        context, ggml_mul(context, second, cosine), ggml_mul(context, first, sine));
    return ggml_concat(context, rotated_first, rotated_second, 0);
}

ggml_tensor * self_attention(
    ggml_context * context,
    detail::GgufModel & model,
    ggml_tensor * value,
    ggml_tensor * cosine,
    ggml_tensor * sine,
    int layer,
    bool flash) {
    constexpr int width = 1024;
    constexpr int heads = 16;
    constexpr int head_width = width / heads;
    const std::string prefix = "layers." + std::to_string(layer) + ".attn.";
    auto project = [&](const char * name) {
        return linear(
            context,
            value,
            named(model, prefix + name + ".weight"),
            named(model, prefix + name + ".bias"));
    };
    auto * query = ggml_reshape_3d(context, project("query_proj"), head_width, heads, value->ne[1]);
    auto * key = ggml_reshape_3d(context, project("key_proj"), head_width, heads, value->ne[1]);
    auto * projected_value = ggml_reshape_3d(context, project("value_proj"), head_width, heads, value->ne[1]);

    query = apply_half_split_rope(context, query, cosine, sine);
    key = apply_half_split_rope(context, key, cosine, sine);

    query = ggml_permute(context, query, 0, 2, 1, 3);
    key = ggml_permute(context, key, 0, 2, 1, 3);
    projected_value = ggml_permute(context, projected_value, 0, 2, 1, 3);
    constexpr float scale = 0.125F;
    ggml_tensor * attended = nullptr;
    if (flash) {
        key = ggml_cast(context, key, GGML_TYPE_F16);
        projected_value = ggml_cast(context, projected_value, GGML_TYPE_F16);
        attended = ggml_flash_attn_ext(context, query, key, projected_value, nullptr, scale, 0.0F, 0.0F);
        ggml_flash_attn_ext_set_prec(attended, GGML_PREC_F32);
    } else {
        attended = attention_f32(context, query, key, projected_value, nullptr, scale);
    }
    attended = ggml_reshape_2d(context, attended, width, value->ne[1]);
    return linear(
        context,
        attended,
        named(model, prefix + "out_proj.weight"),
        named(model, prefix + "out_proj.bias"));
}

ggml_tensor * feed_forward(
    ggml_context * context,
    detail::GgufModel & model,
    ggml_tensor * value,
    int layer,
    const char * module) {
    const std::string prefix = "layers." + std::to_string(layer) + "." + module + ".";
    value = linear(context, value, named(model, prefix + "w_1.weight"), named(model, prefix + "w_1.bias"));
    value = ggml_gelu_erf(context, value);
    return linear(context, value, named(model, prefix + "w_2.weight"), named(model, prefix + "w_2.bias"));
}

ggml_tensor * convolution_module(
    ggml_context * context,
    detail::GgufModel & model,
    ggml_tensor * value,
    int layer) {
    const std::string prefix = "layers." + std::to_string(layer) + ".conv_module.";
    value = layer_norm(
        context,
        value,
        named(model, prefix + "layer_norm.weight"),
        named(model, prefix + "layer_norm.bias"),
        conformer_norm_epsilon);

    auto * first_weight = named(model, prefix + "conv_block.1.weight");
    first_weight = ggml_reshape_2d(context, first_weight, 1024, 2048);
    auto * projected = ggml_mul_mat(context, first_weight, value);
    auto * first = ggml_cont(context, ggml_view_2d(
        context, projected, 1024, projected->ne[1], projected->nb[1], 0));
    auto * gate = ggml_cont(context, ggml_view_2d(
        context, projected, 1024, projected->ne[1], projected->nb[1], 1024 * projected->nb[0]));
    value = ggml_mul(context, first, ggml_sigmoid(context, gate));

    value = conv1d_channels_first(
        context,
        value,
        named(model, prefix + "conv_block.3.weight"),
        nullptr,
        1,
        15,
        true);
    value = layer_norm(
        context,
        value,
        named(model, prefix + "conv_block.4.1.weight"),
        named(model, prefix + "conv_block.4.1.bias"),
        conformer_norm_epsilon);
    value = ggml_gelu_erf(context, value);
    auto * last_weight = named(model, prefix + "conv_block.6.weight");
    last_weight = ggml_reshape_2d(context, last_weight, 1024, 1024);
    return ggml_mul_mat(context, last_weight, value);
}

ggml_tensor * conformer_layer(
    ggml_context * context,
    detail::GgufModel & model,
    ggml_tensor * value,
    ggml_tensor * cosine,
    ggml_tensor * sine,
    int layer,
    bool flash,
    int stop_stage = 5) {
    const std::string prefix = "layers." + std::to_string(layer) + ".";
    auto normalized = [&](const char * module, ggml_tensor * input) {
        return layer_norm(
            context,
            input,
            named(model, prefix + module + ".weight"),
            named(model, prefix + module + ".bias"),
            conformer_norm_epsilon);
    };

    auto * branch = feed_forward(context, model, normalized("ffn1_layer_norm", value), layer, "ffn1");
    value = ggml_add(context, value, ggml_scale(context, branch, 0.5F));
    if (stop_stage == 0) {
        auto * attention_input = normalized("attn_layer_norm", value);
        const std::string query_prefix = "layers." + std::to_string(layer) + ".attn.query_proj.";
        auto * query = linear(
            context,
            attention_input,
            named(model, query_prefix + "weight"),
            named(model, query_prefix + "bias"));
        query = ggml_reshape_3d(context, query, 64, 16, value->ne[1]);
        query = apply_half_split_rope(context, query, cosine, sine);
        return ggml_reshape_2d(context, query, 1024, value->ne[1]);
    }
    if (stop_stage == 1) return value;
    branch = self_attention(context, model, normalized("attn_layer_norm", value), cosine, sine, layer, flash);
    value = ggml_add(context, value, branch);
    if (stop_stage == 2) return value;
    value = ggml_add(context, value, convolution_module(context, model, value, layer));
    if (stop_stage == 3) return value;
    branch = feed_forward(context, model, normalized("ffn2_layer_norm", value), layer, "ffn2");
    value = ggml_add(context, value, ggml_scale(context, branch, 0.5F));
    if (stop_stage == 4) return value;
    return normalized("final_layer_norm", value);
}

} // namespace

class Encoder::Impl {
public:
    Impl(const std::string & path, const EncoderOptions & options)
        : model(detail::load_gguf(path.c_str(), options.device.empty() ? nullptr : options.device.c_str(), options.threads)) {
        ggml_backend_t backends[] = {model.backend(), nullptr};
        std::size_t backend_count = 1;
        auto * primary_device = ggml_backend_get_device(model.backend());
        if (primary_device && ggml_backend_dev_type(primary_device) != GGML_BACKEND_DEVICE_TYPE_CPU) {
            cpu_backend = detail::make_backend("cpu", options.threads, false);
            backends[backend_count++] = cpu_backend;
        }
        scheduler = ggml_backend_sched_new(backends, nullptr, backend_count, graph_size, false, true);
        if (!scheduler) throw std::runtime_error("[yue2:mert2] could not create GGML scheduler");
        std::array<float, 25> logits{};
        ggml_backend_tensor_get(model.get("sheetsage2.layer_weight"), logits.data(), 0, sizeof(logits));
        const float maximum = *std::max_element(logits.begin(), logits.end());
        float total = 0.0F;
        for (std::size_t index = 0; index < logits.size(); ++index) {
            mixture_weights[index] = std::exp(logits[index] - maximum);
            total += mixture_weights[index];
        }
        for (float & weight : mixture_weights) weight /= total;
    }

    ~Impl() {
        if (scheduler) ggml_backend_sched_free(scheduler);
        if (cpu_backend) ggml_backend_free(cpu_backend);
    }

    HiddenFeatures run(
        const LogMelFeatures & features,
        int conformer_layers,
        bool build_sheetsage_memory = false,
        int layer0_stop_stage = 5) {
        if (conformer_layers < 0 || conformer_layers > 24) {
            throw std::invalid_argument("MERT2 layer count must be in [0,24]");
        }
        const bool full_encoder = conformer_layers > 0;
        if (features.bins != 128 || features.frames < 4 ||
            features.values.size() != static_cast<std::size_t>(features.frames * features.bins)) {
            throw std::invalid_argument("MERT2 subsampler expects a non-empty [frames,128] log-mel matrix");
        }

        const std::size_t context_bytes = ggml_tensor_overhead() * graph_size +
            ggml_graph_overhead_custom(graph_size, false);
        std::vector<std::uint8_t> context_storage(context_bytes);
        const ggml_init_params params = {context_bytes, context_storage.data(), true};
        std::unique_ptr<ggml_context, decltype(&ggml_free)> context(ggml_init(params), ggml_free);
        if (!context) throw std::runtime_error("[yue2:mert2] could not allocate graph context");

        auto * graph = ggml_new_graph_custom(context.get(), graph_size, false);
        auto * input = ggml_new_tensor_2d(context.get(), GGML_TYPE_F32, features.bins, features.frames);
        ggml_set_input(input);
        ggml_set_name(input, "mert2.log_mel");
        auto * epsilon = ggml_new_tensor_1d(context.get(), GGML_TYPE_F32, 1);
        ggml_set_input(epsilon);
        ggml_set_name(epsilon, "mert2.grn_epsilon");

        auto * hidden = ggml_div(
            context.get(),
            ggml_sub(context.get(), input, named(model, "feature_extractor.mel_mean")),
            named(model, "feature_extractor.mel_std"));

        constexpr std::array<int, 3> depths = {3, 4, 5};
        constexpr std::array<int, 3> strides = {1, 2, 2};
        for (int block = 0; block < 3; ++block) {
            if (block > 0) {
                const std::string prefix = "sub." + std::to_string(block) + ".resampling_layer";
                hidden = layer_norm(
                    context.get(),
                    hidden,
                    named(model, prefix + ".0.weight"),
                    named(model, prefix + ".0.bias"));
                hidden = conv1d_channels_first(
                    context.get(),
                    hidden,
                    named(model, prefix + ".2.weight"),
                    named(model, prefix + ".2.bias"),
                    strides[static_cast<std::size_t>(block)],
                    0,
                    false);
            }
            for (int layer = 0; layer < depths[static_cast<std::size_t>(block)]; ++layer) {
                hidden = convnext_layer(context.get(), model, hidden, block, layer, epsilon);
            }
        }

        ggml_tensor * rope_cosine = nullptr;
        ggml_tensor * rope_sine = nullptr;
        ggml_tensor * mixed = nullptr;
        std::vector<float> rope_cosine_values;
        std::vector<float> rope_sine_values;
        if (full_encoder) {
            constexpr std::int64_t rotary_half = 32;
            rope_cosine = ggml_new_tensor_3d(context.get(), GGML_TYPE_F32, rotary_half, 1, hidden->ne[1]);
            rope_sine = ggml_new_tensor_3d(context.get(), GGML_TYPE_F32, rotary_half, 1, hidden->ne[1]);
            ggml_set_input(rope_cosine);
            ggml_set_input(rope_sine);
            ggml_set_name(rope_cosine, "mert2.rope_cosine");
            ggml_set_name(rope_sine, "mert2.rope_sine");
            rope_cosine_values.resize(static_cast<std::size_t>(hidden->ne[1] * rotary_half));
            rope_sine_values.resize(rope_cosine_values.size());
            const bool zero_positions = std::getenv("YUE2_ZERO_POSITIONS") != nullptr;
            for (std::int64_t position = 0; position < hidden->ne[1]; ++position) {
                for (std::int64_t index = 0; index < rotary_half; ++index) {
                    const float inverse_frequency = 1.0F / std::pow(10000.0F, static_cast<float>(2 * index) / 64.0F);
                    const float angle = (zero_positions ? 0.0F : static_cast<float>(position)) * inverse_frequency;
                    const auto offset = static_cast<std::size_t>(position * rotary_half + index);
                    rope_cosine_values[offset] = std::cos(angle);
                    rope_sine_values[offset] = std::sin(angle);
                }
            }
            auto * device = ggml_backend_get_device(model.backend());
            const bool flash = hidden->ne[1] > 512 ||
                (device && ggml_backend_dev_type(device) != GGML_BACKEND_DEVICE_TYPE_CPU);
            if (build_sheetsage_memory) {
                if (conformer_layers != 24) throw std::logic_error("SheetSage2 memory requires all 24 layers");
                mixed = ggml_scale(context.get(), hidden, mixture_weights[0]);
            }
            for (int layer = 0; layer < conformer_layers; ++layer) {
                hidden = conformer_layer(
                    context.get(), model, hidden, rope_cosine, rope_sine, layer, flash,
                    layer == 0 ? layer0_stop_stage : 5);
                if (mixed) {
                    mixed = ggml_add(context.get(), mixed,
                        ggml_scale(context.get(), hidden, mixture_weights[static_cast<std::size_t>(layer + 1)]));
                }
            }
            if (mixed) {
                hidden = linear(
                    context.get(),
                    mixed,
                    model.get("sheetsage2.encoder_projection.weight"),
                    model.get("sheetsage2.encoder_projection.bias"));
            }
        }

        ggml_set_name(hidden, build_sheetsage_memory ? "sheetsage2.memory" :
            (full_encoder ? "mert2.encoded" : "mert2.subsampled"));
        ggml_build_forward_expand(graph, hidden);
        ggml_backend_sched_reset(scheduler);
        if (!ggml_backend_sched_alloc_graph(scheduler, graph)) {
            throw std::runtime_error("[yue2:mert2] could not allocate subsampler graph");
        }
        const float epsilon_value = 1.0e-6F;
        ggml_backend_tensor_set(input, features.values.data(), 0, features.values.size() * sizeof(float));
        ggml_backend_tensor_set(epsilon, &epsilon_value, 0, sizeof(epsilon_value));
        if (rope_cosine && layer0_stop_stage >= 2) {
            ggml_backend_tensor_set(
                rope_cosine, rope_cosine_values.data(), 0, rope_cosine_values.size() * sizeof(float));
            ggml_backend_tensor_set(
                rope_sine, rope_sine_values.data(), 0, rope_sine_values.size() * sizeof(float));
        }
        const auto status = ggml_backend_sched_graph_compute(scheduler, graph);
        if (status != GGML_STATUS_SUCCESS) {
            throw std::runtime_error(std::string("[yue2:mert2] graph failed: ") + ggml_status_to_string(status));
        }

        HiddenFeatures result;
        result.channels = hidden->ne[0];
        result.frames = hidden->ne[1];
        result.values.resize(static_cast<std::size_t>(result.channels * result.frames));
        ggml_backend_tensor_get(hidden, result.values.data(), 0, result.values.size() * sizeof(float));
        return result;
    }

    HiddenFeatures subsample(const LogMelFeatures & features) { return run(features, 0); }
    HiddenFeatures encode(const LogMelFeatures & features, int layer_count) { return run(features, layer_count); }
    HiddenFeatures sheetsage_memory(const LogMelFeatures & features) { return run(features, 24, true); }
    HiddenFeatures layer0_stage(const LogMelFeatures & features, int stage) {
        if (stage < 1 || stage > 6) throw std::invalid_argument("MERT2 diagnostic stage must be in [1,6]");
        return run(features, 1, false, stage == 6 ? 0 : stage);
    }

    std::vector<float> decode_logits(
        const HiddenFeatures & memory,
        const std::vector<std::int32_t> & decoder_token_ids) {
        if (memory.channels != 512 || memory.frames <= 0 ||
            memory.values.size() != static_cast<std::size_t>(memory.frames * memory.channels)) {
            throw std::invalid_argument("SheetSage2 decoder expects [frames,512] encoder memory");
        }
        if (decoder_token_ids.empty() || decoder_token_ids.size() > 5120) {
            throw std::invalid_argument("SheetSage2 decoder token count must be in [1,5120]");
        }
        for (const auto token : decoder_token_ids) {
            if (token < 0 || token >= 31678) throw std::invalid_argument("SheetSage2 decoder token is out of range");
        }

        const std::size_t context_bytes = ggml_tensor_overhead() * graph_size +
            ggml_graph_overhead_custom(graph_size, false);
        std::vector<std::uint8_t> context_storage(context_bytes);
        const ggml_init_params params = {context_bytes, context_storage.data(), true};
        std::unique_ptr<ggml_context, decltype(&ggml_free)> context(ggml_init(params), ggml_free);
        if (!context) throw std::runtime_error("[yue2:sheetsage2] could not allocate decoder graph context");
        auto * graph = ggml_new_graph_custom(context.get(), graph_size, false);

        const auto steps = static_cast<std::int64_t>(decoder_token_ids.size());
        auto * memory_tensor = ggml_new_tensor_2d(context.get(), GGML_TYPE_F32, 512, memory.frames);
        auto * token_ids = ggml_new_tensor_1d(context.get(), GGML_TYPE_I32, steps);
        auto * position_ids = ggml_new_tensor_1d(context.get(), GGML_TYPE_I32, steps);
        auto * causal_mask = ggml_new_tensor_2d(context.get(), GGML_TYPE_F32, steps, steps);
        ggml_set_input(memory_tensor);
        ggml_set_input(token_ids);
        ggml_set_input(position_ids);
        ggml_set_input(causal_mask);
        ggml_set_name(memory_tensor, "sheetsage2.memory_input");
        ggml_set_name(token_ids, "sheetsage2.decoder_token_ids");

        auto * hidden = ggml_get_rows(context.get(), model.get("sheetsage2.token_embedding.weight"), token_ids);
        auto * positions = ggml_get_rows(
            context.get(), model.get("sheetsage2.decoder.embed_positions.weight"), position_ids);
        hidden = ggml_add(context.get(), hidden, positions);
        hidden = layer_norm(
            context.get(),
            hidden,
            model.get("sheetsage2.decoder.layernorm_embedding.weight"),
            model.get("sheetsage2.decoder.layernorm_embedding.bias"),
            conformer_norm_epsilon);

        auto * device = ggml_backend_get_device(model.backend());
        const bool flash_cross = memory.frames > 512 ||
            (device && ggml_backend_dev_type(device) != GGML_BACKEND_DEVICE_TYPE_CPU);
        for (int layer = 0; layer < 6; ++layer) {
            hidden = decoder_layer(
                context.get(), model, hidden, memory_tensor, causal_mask, layer, flash_cross);
        }
        auto * logits = ggml_mul_mat(context.get(), model.get("sheetsage2.token_embedding.weight"), hidden);
        ggml_set_name(logits, "sheetsage2.logits");
        ggml_build_forward_expand(graph, logits);

        ggml_backend_sched_reset(scheduler);
        if (!ggml_backend_sched_alloc_graph(scheduler, graph)) {
            throw std::runtime_error("[yue2:sheetsage2] could not allocate decoder graph");
        }
        std::vector<std::int32_t> position_values(decoder_token_ids.size());
        for (std::size_t index = 0; index < position_values.size(); ++index) {
            position_values[index] = static_cast<std::int32_t>(index + 2);
        }
        std::vector<float> mask_values(decoder_token_ids.size() * decoder_token_ids.size());
        for (std::int64_t query = 0; query < steps; ++query) {
            for (std::int64_t key = 0; key < steps; ++key) {
                mask_values[static_cast<std::size_t>(query * steps + key)] =
                    key <= query ? 0.0F : -std::numeric_limits<float>::infinity();
            }
        }
        ggml_backend_tensor_set(
            memory_tensor, memory.values.data(), 0, memory.values.size() * sizeof(float));
        ggml_backend_tensor_set(
            token_ids, decoder_token_ids.data(), 0, decoder_token_ids.size() * sizeof(std::int32_t));
        ggml_backend_tensor_set(
            position_ids, position_values.data(), 0, position_values.size() * sizeof(std::int32_t));
        ggml_backend_tensor_set(
            causal_mask, mask_values.data(), 0, mask_values.size() * sizeof(float));
        const auto status = ggml_backend_sched_graph_compute(scheduler, graph);
        if (status != GGML_STATUS_SUCCESS) {
            throw std::runtime_error(std::string("[yue2:sheetsage2] decoder failed: ") + ggml_status_to_string(status));
        }

        std::vector<float> result(static_cast<std::size_t>(31678 * steps));
        ggml_backend_tensor_get(logits, result.data(), 0, result.size() * sizeof(float));
        return result;
    }

    void prepare_cross_cache(const HiddenFeatures & memory, DecoderCache & cache) {
        const std::size_t context_bytes = ggml_tensor_overhead() * graph_size +
            ggml_graph_overhead_custom(graph_size, false);
        std::vector<std::uint8_t> context_storage(context_bytes);
        const ggml_init_params params = {context_bytes, context_storage.data(), true};
        std::unique_ptr<ggml_context, decltype(&ggml_free)> context(ggml_init(params), ggml_free);
        if (!context) throw std::runtime_error("[yue2:sheetsage2] could not create cross-cache graph");
        auto * graph = ggml_new_graph_custom(context.get(), graph_size, false);
        auto * memory_tensor = ggml_new_tensor_2d(
            context.get(), GGML_TYPE_F32, 512, memory.frames);
        ggml_set_input(memory_tensor);
        ggml_set_name(memory_tensor, "sheetsage2.memory_input");

        for (int layer = 0; layer < 6; ++layer) {
            const std::string prefix = "sheetsage2.decoder.layers." + std::to_string(layer) +
                ".encoder_attn.";
            auto project = [&](const char * name) {
                auto * projected = linear(
                    context.get(), memory_tensor, model.get(prefix + name + ".weight"),
                    model.get(prefix + name + ".bias"));
                projected = ggml_reshape_3d(context.get(), projected, 64, 8, memory.frames);
                projected = ggml_cont(
                    context.get(), ggml_permute(context.get(), projected, 0, 2, 1, 3));
                if (cache.flash_cross) projected = ggml_cast(context.get(), projected, GGML_TYPE_F16);
                return projected;
            };
            ggml_build_forward_expand(graph, ggml_cpy(
                context.get(), project("k_proj"), cache.cross_key[static_cast<std::size_t>(layer)]));
            ggml_build_forward_expand(graph, ggml_cpy(
                context.get(), project("v_proj"), cache.cross_value[static_cast<std::size_t>(layer)]));
        }

        ggml_backend_sched_reset(scheduler);
        if (!ggml_backend_sched_alloc_graph(scheduler, graph)) {
            throw std::runtime_error("[yue2:sheetsage2] could not allocate cross-cache graph");
        }
        ggml_backend_tensor_set(
            memory_tensor, memory.values.data(), 0, memory.values.size() * sizeof(float));
        const auto status = ggml_backend_sched_graph_compute(scheduler, graph);
        if (status != GGML_STATUS_SUCCESS) {
            throw std::runtime_error(std::string("[yue2:sheetsage2] cross-cache projection failed: ") +
                ggml_status_to_string(status));
        }
    }

    std::vector<float> decode_cached_logits(
        DecoderCache & cache,
        const std::vector<std::int32_t> & input_ids,
        std::int64_t cached_tokens) {
        if (input_ids.empty() || cached_tokens < 0 ||
            cached_tokens + static_cast<std::int64_t>(input_ids.size()) > cache.max_tokens) {
            throw std::invalid_argument("SheetSage2 cached decoder position is out of range");
        }
        const std::size_t context_bytes = ggml_tensor_overhead() * graph_size +
            ggml_graph_overhead_custom(graph_size, false);
        std::vector<std::uint8_t> context_storage(context_bytes);
        const ggml_init_params params = {context_bytes, context_storage.data(), true};
        std::unique_ptr<ggml_context, decltype(&ggml_free)> context(ggml_init(params), ggml_free);
        if (!context) throw std::runtime_error("[yue2:sheetsage2] could not create cached decoder graph");
        auto * graph = ggml_new_graph_custom(context.get(), graph_size, false);

        const auto steps = static_cast<std::int64_t>(input_ids.size());
        const auto total_tokens = cached_tokens + steps;
        auto * token_ids = ggml_new_tensor_1d(context.get(), GGML_TYPE_I32, steps);
        auto * position_ids = ggml_new_tensor_1d(context.get(), GGML_TYPE_I32, steps);
        auto * cache_rows = ggml_new_tensor_1d(context.get(), GGML_TYPE_I64, steps);
        auto * causal_mask = ggml_new_tensor_2d(
            context.get(), GGML_TYPE_F32, total_tokens, steps);
        for (auto * input : {token_ids, position_ids, cache_rows, causal_mask}) ggml_set_input(input);
        ggml_set_name(token_ids, "sheetsage2.cached_token_ids");
        ggml_set_name(position_ids, "sheetsage2.cached_position_ids");
        ggml_set_name(cache_rows, "sheetsage2.cache_rows");
        ggml_set_name(causal_mask, "sheetsage2.cached_causal_mask");

        auto * hidden = ggml_get_rows(
            context.get(), model.get("sheetsage2.token_embedding.weight"), token_ids);
        auto * positions = ggml_get_rows(
            context.get(), model.get("sheetsage2.decoder.embed_positions.weight"), position_ids);
        hidden = ggml_add(context.get(), hidden, positions);
        hidden = layer_norm(
            context.get(), hidden, model.get("sheetsage2.decoder.layernorm_embedding.weight"),
            model.get("sheetsage2.decoder.layernorm_embedding.bias"), conformer_norm_epsilon);
        for (int layer = 0; layer < 6; ++layer) {
            hidden = cached_decoder_layer(
                context.get(), graph, model, cache, hidden, causal_mask, cache_rows,
                cached_tokens, layer);
        }
        auto * last_hidden = ggml_view_1d(
            context.get(), hidden, 512,
            static_cast<std::size_t>(steps - 1) * hidden->nb[1]);
        auto * logits = ggml_mul_mat(
            context.get(), model.get("sheetsage2.token_embedding.weight"), last_hidden);
        ggml_set_name(logits, "sheetsage2.cached_logits");
        ggml_build_forward_expand(graph, logits);

        ggml_backend_sched_reset(scheduler);
        if (!ggml_backend_sched_alloc_graph(scheduler, graph)) {
            throw std::runtime_error("[yue2:sheetsage2] could not allocate cached decoder graph");
        }
        std::vector<std::int32_t> positions_data(input_ids.size());
        std::vector<std::int64_t> rows_data(input_ids.size());
        for (std::size_t index = 0; index < input_ids.size(); ++index) {
            positions_data[index] = static_cast<std::int32_t>(cached_tokens + index + 2);
            rows_data[index] = cached_tokens + static_cast<std::int64_t>(index);
        }
        std::vector<float> mask_data(static_cast<std::size_t>(total_tokens * steps));
        for (std::int64_t query = 0; query < steps; ++query) {
            for (std::int64_t key = 0; key < total_tokens; ++key) {
                mask_data[static_cast<std::size_t>(query * total_tokens + key)] =
                    key <= cached_tokens + query ? 0.0F : -std::numeric_limits<float>::infinity();
            }
        }
        ggml_backend_tensor_set(token_ids, input_ids.data(), 0, input_ids.size() * sizeof(std::int32_t));
        ggml_backend_tensor_set(
            position_ids, positions_data.data(), 0, positions_data.size() * sizeof(std::int32_t));
        ggml_backend_tensor_set(
            cache_rows, rows_data.data(), 0, rows_data.size() * sizeof(std::int64_t));
        ggml_backend_tensor_set(
            causal_mask, mask_data.data(), 0, mask_data.size() * sizeof(float));
        const auto status = ggml_backend_sched_graph_compute(scheduler, graph);
        if (status != GGML_STATUS_SUCCESS) {
            throw std::runtime_error(std::string("[yue2:sheetsage2] cached decoder failed: ") +
                ggml_status_to_string(status));
        }
        std::vector<float> result(31678);
        ggml_backend_tensor_get(logits, result.data(), 0, result.size() * sizeof(float));
        return result;
    }

    std::vector<std::int32_t> generate_tokens(
        const HiddenFeatures & memory,
        const std::vector<std::int32_t> & prefix,
        std::size_t max_tokens,
        double stop_time_seconds) {
        using sheetsage2::PromptGrammarState;
        using sheetsage2::TokenLayout;
        if (prefix.empty() || prefix.front() != TokenLayout::sos) {
            throw std::invalid_argument("SheetSage2 generation prefix must begin with SOS");
        }
        const auto out = std::find(prefix.begin(), prefix.end(), TokenLayout::out);
        if (out == prefix.end()) throw std::invalid_argument("SheetSage2 generation prefix must contain OUT");
        if (max_tokens <= prefix.size() || max_tokens > 5120) {
            throw std::invalid_argument("SheetSage2 max token count must exceed the prefix and be at most 5120");
        }
        if (!std::isfinite(stop_time_seconds)) {
            throw std::invalid_argument("SheetSage2 generation stop time must be finite");
        }
        std::vector<std::int32_t> tokens = prefix;
        if (tokens.back() == TokenLayout::eos) tokens.pop_back();
        PromptGrammarState grammar;
        for (auto it = out + 1; it != prefix.end(); ++it) grammar.update(*it);

        auto * device = ggml_backend_get_device(model.backend());
        const bool flash_cross = memory.frames > 512 ||
            (device && ggml_backend_dev_type(device) != GGML_BACKEND_DEVICE_TYPE_CPU);
        DecoderCache cache(
            model.backend(), static_cast<std::int64_t>(max_tokens), memory.frames, flash_cross);
        prepare_cross_cache(memory, cache);
        std::vector<std::int32_t> decoder_input = tokens;
        std::int64_t cached_tokens = 0;

        while (tokens.size() < max_tokens) {
            const auto logits = decode_cached_logits(cache, decoder_input, cached_tokens);
            cached_tokens += static_cast<std::int64_t>(decoder_input.size());
            const auto allowed = grammar.allowed();
            float best_logit = -std::numeric_limits<float>::infinity();
            std::int32_t best_token = -1;
            for (std::int32_t token = 0; token < TokenLayout::vocab_size; ++token) {
                if (!allowed[static_cast<std::size_t>(token)]) continue;
                const float logit = logits[static_cast<std::size_t>(token)];
                if (logit > best_logit) {
                    best_logit = logit;
                    best_token = token;
                }
            }
            if (best_token < 0) throw std::runtime_error("SheetSage2 grammar allowed no decoder token");
            tokens.push_back(best_token);
            if (grammar.update(best_token)) return tokens;
            if (stop_time_seconds >= 0.0 &&
                TokenLayout::type(best_token) == TokenLayout::Type::time &&
                (best_token - TokenLayout::time_begin) / 100.0 >= stop_time_seconds) {
                tokens.push_back(TokenLayout::eos);
                return tokens;
            }
            decoder_input.assign(1, best_token);
        }
        if (tokens.back() != TokenLayout::eos) tokens.push_back(TokenLayout::eos);
        return tokens;
    }

    detail::GgufModel model;
    ggml_backend_t cpu_backend = nullptr;
    ggml_backend_sched_t scheduler = nullptr;
    std::array<float, 25> mixture_weights{};
};

Encoder::Encoder(const std::string & model_path, const EncoderOptions & options)
    : impl_(std::make_unique<Impl>(model_path, options)) {}

Encoder::~Encoder() = default;
Encoder::Encoder(Encoder &&) noexcept = default;
Encoder & Encoder::operator=(Encoder &&) noexcept = default;

HiddenFeatures Encoder::subsample(const LogMelFeatures & features) {
    return impl_->subsample(features);
}

HiddenFeatures Encoder::encode(const LogMelFeatures & features, int layer_count) {
    return impl_->encode(features, layer_count);
}

HiddenFeatures Encoder::sheetsage_memory(const LogMelFeatures & features) {
    return impl_->sheetsage_memory(features);
}

HiddenFeatures Encoder::layer0_stage(const LogMelFeatures & features, int stage) {
    return impl_->layer0_stage(features, stage);
}

std::vector<float> Encoder::decode_logits(
    const HiddenFeatures & memory,
    const std::vector<std::int32_t> & decoder_token_ids) {
    return impl_->decode_logits(memory, decoder_token_ids);
}

std::vector<std::int32_t> Encoder::generate_tokens(
    const HiddenFeatures & memory,
    const std::vector<std::int32_t> & prefix,
    std::size_t max_tokens,
    double stop_time_seconds) {
    return impl_->generate_tokens(memory, prefix, max_tokens, stop_time_seconds);
}

} // namespace yue2::mert2
