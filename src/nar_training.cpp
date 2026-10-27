#include "yue2/nar_training.h"
#include "ggml-alloc.h"
#include "gguf_model.h"
#include "lora.h"
#include "dora.h"
#include <algorithm>
#include <cmath>
#include <filesystem>
#include <limits>
#include <random>
#include <stdexcept>
#include <type_traits>

namespace yue2 {
namespace {
constexpr int H = 2048, D = 128, Q = 16, K = 8, L = 28, FF = 6144;
constexpr size_t C = 4096;
using Context = std::unique_ptr<ggml_context, decltype(&ggml_free)>;
Context context(bool gradients = false) {
    auto bytes = C * ggml_tensor_overhead() + 2 * ggml_graph_overhead_custom(C, gradients);
    Context result(ggml_init({bytes, nullptr, true}), ggml_free);
    if (!result)
        throw std::runtime_error("training graph context allocation failed");
    return result;
}
void finite(const std::vector<float> &values, const char *name) {
    for (auto value : values)
        if (!std::isfinite(value))
            throw std::runtime_error(std::string("nonfinite ") + name);
}
void set(ggml_tensor *t, const std::vector<float> &values) {
    if (ggml_nelements(t) != static_cast<int64_t>(values.size()) || t->type != GGML_TYPE_F32)
        throw std::runtime_error("training tensor upload shape mismatch");
    ggml_backend_tensor_set(t, values.data(), 0, values.size() * sizeof(float));
}
std::vector<float> get(ggml_tensor *t) {
    std::vector<float> values(ggml_nelements(t));
    ggml_backend_tensor_get(t, values.data(), 0, values.size() * sizeof(float));
    return values;
}
ggml_tensor *f32(ggml_context *ctx, ggml_tensor *t) {
    return t->type == GGML_TYPE_F32 ? t : ggml_cast(ctx, t, GGML_TYPE_F32);
}
ggml_tensor *norm(ggml_context *ctx, ggml_tensor *x, ggml_tensor *w) {
    return ggml_mul(ctx, ggml_rms_norm(ctx, x, 1e-6F), f32(ctx, w));
}
bool nar_weight(const std::string &name) {
    return name.find(".nar_") != std::string::npos || name == "model.norm.weight" ||
           name.rfind("vae2llm.", 0) == 0 || name.rfind("llm2vae.", 0) == 0 ||
           name.rfind("time_embedder.", 0) == 0 || name == "latent_pos_embed.pe";
}
struct Graphs {
    Context ctx{nullptr, ggml_free};
    Context forward_ctx{nullptr, ggml_free};
    ggml_cgraph *forward = nullptr, *backward = nullptr;
    ggml_tensor *loss = nullptr;
};
struct Parameter {
    std::string name;
    ggml_tensor *tensor = nullptr;
    std::vector<float> values, gradient, m, v;
};
// Write via a temporary file. A checkpoint is visible only after completion.
void write_file(gguf_context *file, const std::string &path) {
    auto target = std::filesystem::path(path);
    if (!target.parent_path().empty())
        std::filesystem::create_directories(target.parent_path());
    const auto temporary = path + ".partial";
    if (!gguf_write_to_file(file, temporary.c_str(), false))
        throw std::runtime_error("cannot write " + temporary);
    // Checkpoints use unique step names; never silently replace a saved run.
    if (std::filesystem::exists(target)) {
        std::filesystem::remove(temporary);
        throw std::runtime_error("output exists: " + path);
    }
    std::filesystem::rename(temporary, target);
}
} // namespace

class NarTrainer::Impl {
  public:
    Impl(const std::string &path, uint32_t frames, uint32_t max_ar,
         const NarTrainingOptions &options)
        : options(options), frames(frames), n(frames + 2), ar(max_ar),
          model(detail::load_gguf_selected(path.c_str(), options.device.c_str(), options.threads,
                                           nar_weight)),
          fixed(model, options.fixed_adapters) {
        try {
            if (frames < 1 || frames > 2048 || max_ar < 1 || max_ar + frames + 2 > 24576 ||
                options.rank < 1 || options.rank > 128 || !std::isfinite(options.learning_rate) ||
                options.learning_rate <= 0 || !std::isfinite(options.weight_decay) ||
                options.weight_decay < 0 || !std::isfinite(options.gradient_clip) ||
                options.gradient_clip < 0)
                throw std::invalid_argument("invalid NAR training configuration");
            if (model.string("general.architecture") != "yue2" ||
                model.u32("yue2.block_count") != L || model.u32("yue2.embedding_length") != H)
                throw std::runtime_error("unsupported training model");
            if (options.dora && (!options.f32_projections || !options.fixed_adapters.empty()))
                throw std::invalid_argument("DoRA requires F32 projections and no fixed NAR adapters");
            auto require = [&](const std::string &name, int64_t a, int64_t b = 1) {
                auto *t = model.get(name);
                if (t->ne[0] != a || t->ne[1] != b || t->ne[2] != 1 || t->ne[3] != 1)
                    throw std::runtime_error("unexpected NAR training weight shape: " + name);
            };
            for (int layer = 0; layer < L; ++layer) {
                const auto p = "model.layers." + std::to_string(layer) + ".";
                require(p + "nar_input_layernorm.weight", H);
                require(p + "nar_pre_mlp_layernorm.weight", H);
                require(p + "nar_self_attn.q_proj.weight", H, H);
                require(p + "nar_self_attn.k_proj.weight", H, D * K);
                require(p + "nar_self_attn.v_proj.weight", H, D * K);
                require(p + "nar_self_attn.o_proj.weight", H, H);
                require(p + "nar_self_attn.q_norm.weight", D);
                require(p + "nar_self_attn.k_norm.weight", D);
                require(p + "nar_mlp.gate_proj.weight", H, FF);
                require(p + "nar_mlp.up_proj.weight", H, FF);
                require(p + "nar_mlp.down_proj.weight", FF, H);
            }
            require("model.norm.weight", H);
            require("vae2llm.weight", 64, H);
            require("vae2llm.bias", H);
            require("llm2vae.weight", H, 64);
            require("llm2vae.bias", 64);
            require("time_embedder.mlp.0.weight", 256, H);
            require("time_embedder.mlp.0.bias", H);
            require("time_embedder.mlp.2.weight", H, H);
            require("time_embedder.mlp.2.bias", H);
            if (model.get("latent_pos_embed.pe")->ne[0] != H ||
                model.get("latent_pos_embed.pe")->ne[1] < n)
                throw std::runtime_error("invalid latent position table");
            persistent = context();
            std::mt19937 rng(options.seed);
            for (int layer = 0; layer < L; ++layer) {
                const auto base = "model.layers." + std::to_string(layer) + ".";
                for (const auto &suffix :
                     {"nar_self_attn.q_proj", "nar_self_attn.k_proj", "nar_self_attn.v_proj",
                      "nar_self_attn.o_proj", "nar_mlp.gate_proj", "nar_mlp.up_proj",
                      "nar_mlp.down_proj"}) {
                    const auto stem = base + suffix;
                    auto *w = model.get(stem + ".weight");
                    if (options.dora) {
                        auto values = weight_row_norms(w);
                        auto *base = ggml_new_tensor_1d(persistent.get(), GGML_TYPE_F32, w->ne[1]);
                        ggml_set_input(base);
                        base_norms[stem] = base;
                        base_norm_values[stem] = std::move(values);
                    }
                    for (int side = 0; side < (options.dora ? 3 : 2); ++side) {
                        Parameter p;
                        p.name = stem + (side == 0 ? ".lora_A" : side == 1 ? ".lora_B" : ".dora_magnitude");
                        p.tensor = ggml_new_tensor_2d(persistent.get(), GGML_TYPE_F32,
                                                      side == 2 ? w->ne[1] : side ? options.rank : w->ne[0],
                                                      side == 2 ? 1 : side ? w->ne[1] : options.rank);
                        ggml_set_param(p.tensor);
                        ggml_set_input(p.tensor);
                        ggml_set_name(p.tensor, p.name.c_str());
                        p.values.resize(ggml_nelements(p.tensor), 0);
                        if (!side) {
                            std::normal_distribution<float> random(0,
                                                                   1 / std::sqrt(float(w->ne[0])));
                            for (auto &value : p.values)
                                value = random(rng);
                        } else if (side == 2) {
                            const auto &norms = base_norm_values.at(stem);
                            for (size_t i = 0; i < norms.size(); ++i)
                                p.values[i] = std::sqrt(norms[i] + 1e-12F);
                        }
                        p.gradient.resize(p.values.size());
                        p.m.resize(p.values.size());
                        p.v.resize(p.values.size());
                        parameter_index[p.name] = parameters.size();
                        parameters.push_back(std::move(p));
                    }
                }
                keys.push_back(tensor(D, ar, K));
                values.push_back(tensor(D, ar, K));
            }
            for (int layer = 0; layer <= L; ++layer) {
                auto *boundary = tensor(H, n);
                ggml_set_param(boundary);
                boundaries.push_back(boundary);
            }
            mixed = tensor(64, n);
            target = tensor(64, frames);
            features = tensor(256, 1);
            mask = tensor(ar + n, n);
            incoming = tensor(H, n);
            outgoing = tensor(H, n);
            prediction = tensor(64, frames);
            positions = ggml_new_tensor_1d(persistent.get(), GGML_TYPE_I32, n);
            ggml_set_input(positions);
            buffer = ggml_backend_alloc_ctx_tensors(persistent.get(), model.backend());
            if (!buffer)
                throw std::runtime_error("cannot allocate training state");
            for (auto &p : parameters)
                set(p.tensor, p.values);
            for (const auto &entry : base_norms)
                set(entry.second, base_norm_values.at(entry.first));
            build_front();
            build_tail();
            for (int layer = 0; layer < L; ++layer)
                build_block(layer);
            std::map<std::string, ggml_gallocr_t> shapes;
            for (auto *graph : graphs()) {
                std::string shape;
                for (int i = 0; i < ggml_graph_n_nodes(graph); ++i) {
                    auto *node = ggml_graph_node(graph, i);
                    if (options.f32_projections && node->op == GGML_OP_MUL_MAT)
                        ggml_mul_mat_set_prec(node, GGML_PREC_F32);
                    if (node->op != GGML_OP_NONE &&
                        !ggml_backend_supports_op(model.backend(), node))
                        throw std::runtime_error(std::string("training backend lacks ") +
                                                 ggml_op_name(node->op));
                    shape += std::to_string(node->op) + ":" + std::to_string(node->type);
                    for (auto extent : node->ne)
                        shape += "," + std::to_string(extent);
                    for (auto *src : node->src) {
                        if (!src) {
                            shape += ";null";
                            continue;
                        }
                        shape += ";" + std::to_string(src->op) + ":" + std::to_string(src->type) +
                                 ":" + std::to_string(src->flags);
                        for (auto extent : src->ne)
                            shape += "," + std::to_string(extent);
                    }
                    shape += '|';
                }
                auto found = shapes.find(shape);
                if (found == shapes.end()) {
                    auto workspace =
                        ggml_gallocr_new(ggml_backend_get_default_buffer_type(model.backend()));
                    if (!workspace)
                        throw std::runtime_error("cannot create training allocator");
                    workspaces.push_back(workspace);
                    if (!ggml_gallocr_reserve(workspace, graph))
                        throw std::runtime_error("cannot reserve training graph");
                    found = shapes.emplace(std::move(shape), workspace).first;
                }
                graph_allocators[graph] = found->second;
            }
            double bytes = 0;
            for (auto workspace : workspaces)
                bytes += ggml_gallocr_get_buffer_size(workspace, 0);
            std::fprintf(stderr,
                         "[yue2:train] %u frames, %u AR slots, rank %d, 28 layers, "
                         "%zu workspace "
                         "shapes %.1f MiB\n",
                         frames, ar, options.rank, workspaces.size(), bytes / 1048576.0);
        } catch (...) {
            for (auto workspace : workspaces)
                ggml_gallocr_free(workspace);
            if (buffer)
                ggml_backend_buffer_free(buffer);
            throw;
        }
    }
    ~Impl() {
        for (auto workspace : workspaces)
            ggml_gallocr_free(workspace);
        if (buffer)
            ggml_backend_buffer_free(buffer);
    }
    ggml_tensor *tensor(int64_t a, int64_t b, int64_t c = 1) {
        auto *t = ggml_new_tensor_3d(persistent.get(), GGML_TYPE_F32, a, b, c);
        ggml_set_input(t);
        return t;
    }
    static std::vector<float> weight_row_norms(ggml_tensor *w) {
        const auto count = ggml_nelements(w);
        std::vector<float> values(count);
        if (w->type == GGML_TYPE_F32) {
            ggml_backend_tensor_get(w, values.data(), 0, values.size() * sizeof(float));
        } else if (w->type == GGML_TYPE_F16) {
            std::vector<ggml_fp16_t> half(count);
            ggml_backend_tensor_get(w, half.data(), 0, half.size() * sizeof(ggml_fp16_t));
            ggml_fp16_to_fp32_row(half.data(), values.data(), count);
        } else {
            throw std::invalid_argument("DoRA training requires F16/F32 frozen weights");
        }
        std::vector<float> result(w->ne[1]);
        for (int64_t row = 0; row < w->ne[1]; ++row) {
            double sum = 0;
            for (int64_t i = 0; i < w->ne[0]; ++i) {
                const auto v = values[row * w->ne[0] + i]; sum += double(v) * v;
            }
            if (!std::isfinite(sum) || sum <= 1e-12)
                throw std::runtime_error("DoRA base has a zero/nonfinite output row");
            result[row] = static_cast<float>(sum);
        }
        return result;
    }
    int parameters_per_layer() const { return options.dora ? 21 : 14; }
    ggml_tensor *linear(ggml_context *ctx, ggml_tensor *x, const std::string &name,
                        bool train = true) {
        auto *output = fixed.linear(ctx, model.get(name + ".weight"), x, options.f32_projections);
        auto found = parameter_index.find(name + ".lora_A");
        if (train && found != parameter_index.end()) {
            auto *a = parameters[found->second].tensor;
            auto *b = parameters[found->second + 1].tensor;
            output = ggml_add(ctx, output, ggml_mul_mat(ctx, b, ggml_mul_mat(ctx, a, x)));
            if (options.dora) {
                auto *denom = detail::dora_row_norm(ctx, f32(ctx, model.get(name + ".weight")),
                                                   a, b, base_norms.at(name));
                auto *magnitude = parameters[found->second + 2].tensor;
                output = ggml_mul(ctx, output,
                    ggml_div(ctx, ggml_reshape_1d(ctx, magnitude, magnitude->ne[0]), denom));
            }
        }
        return output;
    }
    ggml_tensor *bias(ggml_context *ctx, ggml_tensor *x, const std::string &name) {
        return ggml_add(ctx, linear(ctx, x, name),
                        f32(ctx, fixed.bias(ctx, model.get(name + ".bias"))));
    }
    ggml_tensor *block(ggml_context *ctx, ggml_tensor *x, int layer) {
        const auto base = "model.layers." + std::to_string(layer) + ".";
        auto *normalized = norm(ctx, x, model.get(base + "nar_input_layernorm.weight"));
        const auto attn = base + "nar_self_attn.";
        auto *q = ggml_reshape_3d(ctx, linear(ctx, normalized, attn + "q_proj"), D, Q, n);
        auto *k = ggml_reshape_3d(ctx, linear(ctx, normalized, attn + "k_proj"), D, K, n);
        auto *v = ggml_reshape_3d(ctx, linear(ctx, normalized, attn + "v_proj"), D, K, n);
        q = norm(ctx, q, model.get(attn + "q_norm.weight"));
        k = norm(ctx, k, model.get(attn + "k_norm.weight"));
        q = ggml_rope_ext(ctx, q, positions, nullptr, D, GGML_ROPE_TYPE_NEOX, 0, 1e6F, 1, 0, 1, 0,
                          0);
        k = ggml_rope_ext(ctx, k, positions, nullptr, D, GGML_ROPE_TYPE_NEOX, 0, 1e6F, 1, 0, 1, 0,
                          0);
        q = ggml_permute(ctx, q, 0, 2, 1, 3);
        k = ggml_concat(ctx, keys[layer], ggml_cont(ctx, ggml_permute(ctx, k, 0, 2, 1, 3)), 1);
        v = ggml_concat(ctx, values[layer], ggml_cont(ctx, ggml_permute(ctx, v, 0, 2, 1, 3)), 1);
        auto *scores =
            ggml_soft_max_ext(ctx, ggml_mul_mat(ctx, k, q), mask, 1 / std::sqrt(float(D)), 0);
        auto *attended = ggml_mul_mat(ctx, ggml_cont(ctx, ggml_transpose(ctx, v)), scores);
        attended =
            ggml_reshape_2d(ctx, ggml_cont(ctx, ggml_permute(ctx, attended, 0, 2, 1, 3)), H, n);
        x = ggml_add(ctx, x, linear(ctx, attended, attn + "o_proj"));
        normalized = norm(ctx, x, model.get(base + "nar_pre_mlp_layernorm.weight"));
        auto *gate = linear(ctx, normalized, base + "nar_mlp.gate_proj");
        auto *up = linear(ctx, normalized, base + "nar_mlp.up_proj");
        return ggml_add(ctx, x,
                        linear(ctx, ggml_swiglu_split(ctx, gate, up), base + "nar_mlp.down_proj"));
    }
    void build_front() {
        front.ctx = context();
        auto *ctx = front.ctx.get();
        auto *hidden = bias(ctx, mixed, "vae2llm");
        auto *time = bias(ctx, ggml_silu(ctx, bias(ctx, features, "time_embedder.mlp.0")),
                          "time_embedder.mlp.2");
        auto *table = model.get("latent_pos_embed.pe");
        hidden = ggml_add(ctx, ggml_add(ctx, hidden, time),
                          f32(ctx, ggml_view_2d(ctx, table, H, n, table->nb[1], 0)));
        front.forward = ggml_new_graph_custom(ctx, C, false);
        ggml_build_forward_expand(front.forward, ggml_cpy(ctx, hidden, boundaries[0]));
    }
    void build_tail() {
        tail.ctx = context(true);
        auto *ctx = tail.ctx.get();
        auto *hidden = norm(ctx, boundaries.back(), model.get("model.norm.weight"));
        auto *out = bias(ctx, hidden, "llm2vae");
        auto *velocity = ggml_cont(ctx, ggml_view_2d(ctx, out, 64, frames, out->nb[1], out->nb[1]));
        auto *difference = ggml_sub(ctx, velocity, target);
        tail.loss = ggml_scale(ctx, ggml_sum(ctx, ggml_sqr(ctx, difference)), 1.0F / (64 * frames));
        ggml_set_loss(tail.loss);
        ggml_set_output(tail.loss);
        tail.backward = ggml_new_graph_custom(ctx, C, true);
        ggml_build_forward_expand(tail.backward, ggml_cpy(ctx, velocity, prediction));
        ggml_build_forward_expand(tail.backward, tail.loss);
        ggml_build_backward_expand(ctx, tail.backward, nullptr);
        ggml_build_forward_expand(
            tail.backward,
            ggml_cpy(ctx, ggml_graph_get_grad(tail.backward, boundaries.back()), incoming));
    }
    void build_block(int layer) {
        blocks.emplace_back();
        auto &graph = blocks.back();
        graph.forward_ctx = context();
        auto *ctx = graph.forward_ctx.get();
        auto *hidden = block(ctx, boundaries[layer], layer);
        graph.forward = ggml_new_graph_custom(ctx, C, false);
        ggml_build_forward_expand(graph.forward, ggml_cpy(ctx, hidden, boundaries[layer + 1]));
        // Separate graph tensors are essential: forward allocation can reuse
        // activations whose values are needed by the recomputed backward graph.
        graph.ctx = context(true);
        ctx = graph.ctx.get();
        hidden = block(ctx, boundaries[layer], layer);
        graph.loss = ggml_sum(ctx, ggml_mul(ctx, hidden, incoming));
        ggml_set_loss(graph.loss);
        graph.backward = ggml_new_graph_custom(ctx, C, true);
        ggml_build_forward_expand(graph.backward, graph.loss);
        ggml_build_backward_expand(ctx, graph.backward, nullptr);
        ggml_build_forward_expand(
            graph.backward,
            ggml_cpy(ctx, ggml_graph_get_grad(graph.backward, boundaries[layer]), outgoing));
        for (int i = layer * parameters_per_layer(); i < (layer + 1) * parameters_per_layer(); ++i) {
            auto *gradient = ggml_graph_get_grad(graph.backward, parameters[i].tensor);
            if (!gradient)
                throw std::runtime_error("missing adapter gradient");
            ggml_set_output(gradient);
        }
    }
    std::vector<ggml_cgraph *> graphs() {
        std::vector<ggml_cgraph *> result{front.forward, tail.backward};
        for (auto &b : blocks) {
            result.push_back(b.forward);
            result.push_back(b.backward);
        }
        return result;
    }
    void compute(ggml_cgraph *graph, bool backward = false) {
        if (!ggml_gallocr_alloc_graph(graph_allocators.at(graph), graph))
            throw std::runtime_error("training graph allocation failed");
        if (backward)
            ggml_graph_reset(graph);
        if (ggml_backend_graph_compute(model.backend(), graph) != GGML_STATUS_SUCCESS)
            throw std::runtime_error("training compute failed");
    }
    void inputs(const NarTrainingSample &sample, const std::vector<float> &noise, float sigma) {
        const auto count = sample.conditioning.token_count;
        if (sample.clean_latents.size() != frames * 64ULL || noise.size() != frames * 64ULL ||
            count < 1 || count > ar ||
            sample.conditioning.keys.size() != L * K * D * static_cast<size_t>(count) ||
            sample.conditioning.values.size() != sample.conditioning.keys.size() ||
            !std::isfinite(sigma) || sigma <= 0 || sigma >= 1)
            throw std::invalid_argument("invalid NAR training sample");
        finite(sample.clean_latents, "clean latents");
        finite(noise, "noise");
        finite(sample.conditioning.keys, "AR keys");
        finite(sample.conditioning.values, "AR values");
        for (int layer = 0; layer < L; ++layer) {
            std::vector<float> pad(ar * K * D, 0);
            for (int which = 0; which < 2; ++which) {
                const auto &data = which ? sample.conditioning.values : sample.conditioning.keys;
                for (int head = 0; head < K; ++head)
                    std::copy_n(data.data() + (layer * K + head) * count * D, count * D,
                                pad.data() + head * ar * D);
                set(which ? values[layer] : keys[layer], pad);
            }
        }
        std::vector<int32_t> pos(n);
        for (uint32_t i = 0; i < n; ++i)
            pos[i] = count + i;
        ggml_backend_tensor_set(positions, pos.data(), 0, pos.size() * sizeof(pos[0]));
        std::vector<float> attention_mask((ar + n) * n, 0);
        for (uint32_t i = 0; i < n; ++i)
            for (uint32_t j = count; j < ar; ++j)
                attention_mask[i * (ar + n) + j] = -std::numeric_limits<float>::infinity();
        set(mask, attention_mask);
        std::vector<float> x(n * 64, 0), v(frames * 64), t(256);
        for (size_t i = 0; i < v.size(); ++i) {
            x[i + 64] = (1 - sigma) * sample.clean_latents[i] + sigma * noise[i];
            v[i] = noise[i] - sample.clean_latents[i];
        }
        for (int i = 0; i < 128; ++i) {
            const auto angle = sigma * std::exp(-std::log(10000.0F) * i / 128.0F);
            t[i] = std::cos(angle);
            t[i + 128] = std::sin(angle);
        }
        set(mixed, x);
        set(target, v);
        set(features, t);
        last_sigma = sigma;
        last_count = count;
    }
    float forward() {
        compute(front.forward);
        for (auto &graph : blocks)
            compute(graph.forward);
        compute(tail.backward, true);
        const auto loss = get(tail.loss)[0];
        if (!std::isfinite(loss))
            throw std::runtime_error("nonfinite training loss");
        return loss;
    }
    NarTrainingResult step(const NarTrainingSample &sample, const std::vector<float> &noise,
                           float sigma, bool update) {
        inputs(sample, noise, sigma);
        const float loss = forward();
        double squared = 0;
        for (int layer = L - 1; layer >= 0; --layer) {
            auto &graph = blocks[layer];
            compute(graph.backward, true);
            for (int i = layer * parameters_per_layer(); i < (layer + 1) * parameters_per_layer(); ++i) {
                auto &p = parameters[i];
                p.gradient = get(ggml_graph_get_grad(graph.backward, p.tensor));
                finite(p.gradient, "adapter gradient");
                for (auto value : p.gradient)
                    squared += double(value) * value;
            }
            ggml_backend_tensor_copy(outgoing, incoming);
        }
        const auto gradient_norm = std::sqrt(squared);
        if (!std::isfinite(gradient_norm))
            throw std::runtime_error("nonfinite global gradient norm");
        if (update) {
            ++optimizer_step;
            const double scale = options.gradient_clip > 0 && gradient_norm > options.gradient_clip
                                     ? options.gradient_clip / gradient_norm
                                     : 1;
            const double correction1 = 1 - std::pow(0.9, double(optimizer_step));
            const double correction2 = 1 - std::pow(0.999, double(optimizer_step));
            for (auto &p : parameters) {
                for (size_t i = 0; i < p.values.size(); ++i) {
                    const float g = static_cast<float>(p.gradient[i] * scale);
                    p.m[i] = 0.9F * p.m[i] + 0.1F * g;
                    p.v[i] = 0.999F * p.v[i] + 0.001F * g * g;
                    p.values[i] -= options.learning_rate *
                                   static_cast<float>((p.m[i] / correction1) /
                                                          (std::sqrt(p.v[i] / correction2) + 1e-8) +
                                                      options.weight_decay * p.values[i]);
                }
                finite(p.values, "updated adapter");
                set(p.tensor, p.values);
            }
        }
        return {loss, gradient_norm, optimizer_step};
    }
    void metadata(gguf_context *file, bool state = false) const {
        gguf_set_val_str(file, "general.architecture", state ? "yue2_nar_training" : "yue2_lora");
        gguf_set_val_str(file, "yue2.component",
                         state ? "nar-training-state" : "generation-adapter");
        gguf_set_val_str(file, "yue2.adapter.type", options.dora ? "dora" : "lora");
        if (options.dora) {
            gguf_set_val_str(file, "yue2.adapter.dora.norm_gradient", "full");
            gguf_set_val_str(file, "yue2.adapter.dora.scale_policy", "blend complete normalized delta");
            gguf_set_val_u32(file, "yue2.adapter.dora.target_count", L * 7);
        }
        gguf_set_val_u32(file, "yue2.adapter.rank", options.rank);
        gguf_set_val_f32(file, "yue2.adapter.alpha", float(options.rank));
        gguf_set_val_u32(file, "yue2.adapter.target_count", L * 7);
        if (auto sha = model.optional_string("yue2.checkpoint.sha256"))
            gguf_set_val_str(file, "yue2.adapter.base_sha256", sha->c_str());
        gguf_set_val_str(file, "yue2.training.identity", options.run_identity.c_str());
        gguf_set_val_str(file, "yue2.training.fixed_adapters", fixed_identity().c_str());
        gguf_set_val_str(file, "yue2.training.conditioning_adapters",
                         conditioning_identity().c_str());
        gguf_set_val_str(file, "yue2.training.f32_projections",
                         options.f32_projections ? "1" : "0");
        gguf_set_val_u64(file, "yue2.training.step", optimizer_step);
        gguf_set_val_str(file, "yue2.training.step_text", std::to_string(optimizer_step).c_str());
        gguf_set_val_u32(file, "yue2.training.frames", frames);
        gguf_set_val_u32(file, "yue2.training.ar_slots", ar);
        gguf_set_val_f32(file, "yue2.training.learning_rate", options.learning_rate);
        gguf_set_val_f32(file, "yue2.training.weight_decay", options.weight_decay);
        gguf_set_val_f32(file, "yue2.training.gradient_clip", options.gradient_clip);
        gguf_set_val_str(file, "yue2.training.composition",
                         "delta stacked after fixed adapters; keep the fixed "
                         "companion when rendering");
    }
    std::string fixed_identity() const {
        std::string result;
        for (const auto &spec : options.fixed_adapters)
            result += std::filesystem::absolute(spec.path).lexically_normal().string() + "=" +
                      std::to_string(spec.strength) + "\n";
        return result;
    }
    std::string conditioning_identity() const {
        std::string result;
        for (const auto &spec : options.conditioning_adapters)
            result += std::filesystem::absolute(spec.path).lexically_normal().string() + "=" +
                      std::to_string(spec.strength) + "\n";
        return result;
    }
    std::map<std::string, std::vector<float>> export_row_gains() const {
        std::map<std::string, std::vector<float>> result;
        for (const auto &entry : base_norms) {
            auto ctx = context();
            const auto at = parameter_index.at(entry.first + ".lora_A");
            auto *denom = detail::dora_row_norm(ctx.get(), f32(ctx.get(), model.get(entry.first + ".weight")),
                parameters[at].tensor, parameters[at + 1].tensor, entry.second);
            auto *graph = ggml_new_graph_custom(ctx.get(), C, false);
            ggml_set_output(denom); ggml_build_forward_expand(graph, denom);
            for (int i = 0; i < ggml_graph_n_nodes(graph); ++i) {
                auto *node = ggml_graph_node(graph, i);
                if (node->op == GGML_OP_MUL_MAT) ggml_mul_mat_set_prec(node, GGML_PREC_F32);
            }
            auto *allocator = ggml_gallocr_new(ggml_backend_get_default_buffer_type(model.backend()));
            if (!allocator) throw std::runtime_error("cannot allocate DoRA export graph");
            std::unique_ptr<std::remove_pointer_t<ggml_gallocr_t>, decltype(&ggml_gallocr_free)> guard(allocator, ggml_gallocr_free);
            if (!ggml_gallocr_alloc_graph(allocator, graph) ||
                ggml_backend_graph_compute(model.backend(), graph) != GGML_STATUS_SUCCESS)
                throw std::runtime_error("DoRA row-norm export failed");
            auto norms = get(denom);
            const auto &m = parameters[at + 2].values;
            const auto &base = base_norm_values.at(entry.first);
            for (size_t i = 0; i < norms.size(); ++i) {
                if (!std::isfinite(norms[i]) || norms[i] * norms[i] < 1e-8F * base[i])
                    throw std::runtime_error("DoRA direction nearly cancels a base row; unsafe export");
                norms[i] = m[i] / norms[i];
            }
            finite(norms, "DoRA row gains");
            result[entry.first] = std::move(norms);
        }
        return result;
    }
    void write(const std::string &path, bool state, const std::string &rng = "",
               bool diagnostics = false) const {
        Context ctx = context();
        auto *file = gguf_init_empty();
        std::unique_ptr<gguf_context, decltype(&gguf_free)> guard(file, gguf_free);
        metadata(file, state || diagnostics);
        auto add = [&](const std::string &name, const std::vector<float> &data, int64_t a,
                       int64_t b = 1, int64_t c = 1) {
            auto *t = ggml_new_tensor_3d(ctx.get(), GGML_TYPE_F32, a, b, c);
            if (ggml_nelements(t) != static_cast<int64_t>(data.size()))
                throw std::runtime_error("checkpoint shape mismatch");
            ggml_set_name(t, name.c_str());
            t->data = const_cast<float *>(data.data());
            gguf_add_tensor(file, t);
        };
        // Diagnostic tensors need stable host storage until the file is written.
        std::vector<std::vector<float>> extra;
        extra.reserve(L * 2 + 4);
        for (const auto &p : parameters) {
            add(p.name, p.values, p.tensor->ne[0], p.tensor->ne[1]);
            if (state) {
                add(p.name + ".moment1", p.m, p.tensor->ne[0], p.tensor->ne[1]);
                add(p.name + ".moment2", p.v, p.tensor->ne[0], p.tensor->ne[1]);
            }
            if (diagnostics)
                add(p.name + ".grad", p.gradient, p.tensor->ne[0], p.tensor->ne[1]);
        }
        const auto gains = options.dora && !state && !diagnostics ? export_row_gains()
                                                                 : std::map<std::string, std::vector<float>>{};
        for (const auto &entry : gains)
            add(entry.first + ".dora_scale", entry.second, entry.second.size());
        if (diagnostics) {
            gguf_set_val_f32(file, "yue2.training.sigma", last_sigma);
            gguf_set_val_u32(file, "yue2.training.actual_ar_tokens", last_count);
            auto snapshot = [&](const std::string &name, ggml_tensor *t) {
                extra.push_back(get(t));
                add(name, extra.back(), t->ne[0], t->ne[1], t->ne[2]);
            };
            snapshot("mixed", mixed);
            snapshot("target", target);
            snapshot("velocity", prediction);
            for (int i = 0; i < L; ++i) {
                snapshot("key." + std::to_string(i), keys[i]);
                snapshot("value." + std::to_string(i), values[i]);
            }
        }
        if (state)
            gguf_set_val_str(file, "yue2.training.rng", rng.c_str());
        write_file(file, path);
    }
    std::string load(const std::string &path) {
        auto saved = detail::load_gguf_raw(path.c_str(), "cpu", 1);
        if (saved.string("general.architecture") != "yue2_nar_training" ||
            saved.string("yue2.adapter.type") != (options.dora ? "dora" : "lora") ||
            (options.dora && saved.optional_string("yue2.adapter.dora.norm_gradient").value_or("") != "full") ||
            saved.u32("yue2.adapter.rank") != static_cast<uint32_t>(options.rank) ||
            saved.u32("yue2.training.frames") != frames ||
            saved.u32("yue2.training.ar_slots") != ar ||
            saved.string("yue2.training.identity") != options.run_identity ||
            saved.f32("yue2.training.learning_rate") != options.learning_rate ||
            saved.string("yue2.training.fixed_adapters") != fixed_identity() ||
            saved.optional_string("yue2.training.conditioning_adapters").value_or("") !=
                conditioning_identity() ||
            saved.optional_string("yue2.training.f32_projections").value_or("0") !=
                (options.f32_projections ? "1" : "0") ||
            saved.f32("yue2.training.weight_decay") != options.weight_decay ||
            saved.f32("yue2.training.gradient_clip") != options.gradient_clip ||
            saved.optional_string("yue2.adapter.base_sha256") !=
                model.optional_string("yue2.checkpoint.sha256"))
            throw std::runtime_error("checkpoint does not match training configuration");
        // Validate all tensors before applying any state.
        for (const auto &p : parameters)
            for (const auto &suffix : {"", ".moment1", ".moment2"}) {
                auto *t = saved.get(p.name + suffix);
                if (t->type != GGML_TYPE_F32 || !ggml_are_same_shape(t, p.tensor))
                    throw std::runtime_error("invalid optimizer state shape");
                finite(get(t), "saved optimizer state");
                if (std::string(suffix) == ".moment2")
                    for (auto value : get(t))
                        if (value < 0)
                            throw std::runtime_error("negative saved AdamW variance");
            }
        const auto step = std::stoull(saved.string("yue2.training.step_text"));
        const auto rng = saved.string("yue2.training.rng");
        for (auto &p : parameters) {
            p.values = get(saved.get(p.name));
            p.m = get(saved.get(p.name + ".moment1"));
            p.v = get(saved.get(p.name + ".moment2"));
            set(p.tensor, p.values);
        }
        optimizer_step = step;
        return rng;
    }
    NarTrainingOptions options;
    uint32_t frames, n, ar, last_count = 0;
    float last_sigma = 0;
    detail::GgufModel model;
    detail::LoraStack fixed;
    Context persistent{nullptr, ggml_free};
    ggml_backend_buffer_t buffer = nullptr;
    std::vector<ggml_gallocr_t> workspaces;
    std::map<ggml_cgraph *, ggml_gallocr_t> graph_allocators;
    std::vector<Parameter> parameters;
    std::map<std::string, size_t> parameter_index;
    std::map<std::string, ggml_tensor *> base_norms;
    std::map<std::string, std::vector<float>> base_norm_values;
    std::vector<ggml_tensor *> keys, values, boundaries;
    ggml_tensor *mixed = nullptr, *target = nullptr, *features = nullptr, *mask = nullptr,
                *positions = nullptr;
    ggml_tensor *incoming = nullptr, *outgoing = nullptr, *prediction = nullptr;
    Graphs front, tail;
    std::vector<Graphs> blocks;
    uint64_t optimizer_step = 0;
};

NarTrainer::NarTrainer(const std::string &model, uint32_t frames, uint32_t ar,
                       const NarTrainingOptions &options)
    : impl_(std::make_unique<Impl>(model, frames, ar, options)) {}
NarTrainer::~NarTrainer() = default;
NarTrainingResult NarTrainer::step(const NarTrainingSample &s, const std::vector<float> &noise,
                                   float sigma, bool update) {
    return impl_->step(s, noise, sigma, update);
}
float NarTrainer::evaluate(const NarTrainingSample &s, const std::vector<float> &noise,
                           float sigma) {
    impl_->inputs(s, noise, sigma);
    return impl_->forward();
}
std::vector<float> NarTrainer::velocity() const { return get(impl_->prediction); }
void NarTrainer::save_adapter(const std::string &path) const { impl_->write(path, false); }
void NarTrainer::save_state(const std::string &path, const std::string &rng) const {
    impl_->write(path, true, rng);
}
std::string NarTrainer::load_state(const std::string &path) { return impl_->load(path); }
void NarTrainer::save_diagnostics(const std::string &path) const {
    impl_->write(path, false, "", true);
}
} // namespace yue2
