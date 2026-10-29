#include "yue2/ar_training.h"
#include "ggml-alloc.h"
#include "gguf_model.h"
#include "lora.h"
#include "dora.h"
#include "training_parallel.h"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <future>
#include <limits>
#include <random>
#include <stdexcept>
#include <type_traits>

namespace yue2 {
namespace {
constexpr int H = 2048, D = 128, Q = 16, K = 8, L = 28, FF = 6144;
constexpr size_t C = 4096;
using Context = std::unique_ptr<ggml_context, decltype(&ggml_free)>;
using Clock = std::chrono::steady_clock;
double since(Clock::time_point start) {
    return std::chrono::duration<double>(Clock::now() - start).count();
}
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
// Download into existing host storage, avoiding a fresh allocation per step.
void download(ggml_tensor *t, std::vector<float> &values) {
    if (ggml_nelements(t) != static_cast<int64_t>(values.size()) || t->type != GGML_TYPE_F32)
        throw std::runtime_error("training tensor download shape mismatch");
    ggml_backend_tensor_get(t, values.data(), 0, values.size() * sizeof(float));
}
ggml_tensor *f32(ggml_context *ctx, ggml_tensor *t) {
    return t->type == GGML_TYPE_F32 ? t : ggml_cast(ctx, t, GGML_TYPE_F32);
}
ggml_tensor *norm(ggml_context *ctx, ggml_tensor *x, ggml_tensor *w) {
    return ggml_mul(ctx, ggml_rms_norm(ctx, x, 1e-6F), f32(ctx, w));
}
bool ar_weight(const std::string &name) {
    return name == "model.embed_tokens.weight" || name == "lm_head.weight" ||
           name == "model.norm.weight" ||
           (name.rfind("model.layers.", 0) == 0 && name.find(".nar_") == std::string::npos);
}
struct Graphs {
    Context ctx{nullptr, ggml_free};
    Context forward_ctx{nullptr, ggml_free};
    Context teacher_ctx{nullptr, ggml_free};
    ggml_cgraph *teacher_forward = nullptr;
    ggml_cgraph *forward = nullptr, *backward = nullptr;
    ggml_tensor *loss = nullptr, *task_loss = nullptr, *kl_loss = nullptr;
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

class ArTrainer::Impl {
  public:
    Impl(const std::string &path, uint32_t prefix_tokens, uint32_t frames, bool predict_end,
         const ArTrainingOptions &options)
        : options(options), frames(frames), ar(prefix_tokens),
          n(prefix_tokens + frames - (predict_end ? 0 : 1)),
          targets(frames + (predict_end ? 1 : 0)), predict_end(predict_end),
          model(detail::load_gguf_selected(path.c_str(), options.device.c_str(), options.threads,
                                           ar_weight)),
          fixed(model, options.fixed_adapters) {
        try {
            if (frames < 1 || frames > 2048 || ar < 1 || n > 4096 || options.rank < 1 ||
                options.rank > 128 || !std::isfinite(options.learning_rate) ||
                options.learning_rate <= 0 || !std::isfinite(options.weight_decay) ||
                options.weight_decay < 0 || !std::isfinite(options.gradient_clip) ||
                options.gradient_clip < 0 || !std::isfinite(options.kl_weight) ||
                options.kl_weight < 0)
                throw std::invalid_argument("invalid AR training configuration");
            if (model.string("general.architecture") != "yue2" ||
                model.u32("yue2.block_count") != L || model.u32("yue2.embedding_length") != H)
                throw std::runtime_error("unsupported training model");
            if(options.dora && fixed.has_non_additive_targets())
                throw std::runtime_error("AR DoRA currently requires ordinary additive frozen companions");
            if(options.dora) reference_id=fixed.reference_signature();
            auto require = [&](const std::string &name, int64_t a, int64_t b = 1) {
                auto *t = model.get(name);
                if (t->ne[0] != a || t->ne[1] != b || t->ne[2] != 1 || t->ne[3] != 1)
                    throw std::runtime_error("unexpected AR training weight shape: " + name);
            };
            for (int layer = 0; layer < L; ++layer) {
                const auto p = "model.layers." + std::to_string(layer) + ".";
                require(p + "input_layernorm.weight", H);
                require(p + "post_attention_layernorm.weight", H);
                require(p + "self_attn.q_proj.weight", H, H);
                require(p + "self_attn.k_proj.weight", H, D * K);
                require(p + "self_attn.v_proj.weight", H, D * K);
                require(p + "self_attn.o_proj.weight", H, H);
                require(p + "self_attn.q_norm.weight", D);
                require(p + "self_attn.k_norm.weight", D);
                require(p + "mlp.gate_proj.weight", H, FF);
                require(p + "mlp.up_proj.weight", H, FF);
                require(p + "mlp.down_proj.weight", FF, H);
            }
            require("model.norm.weight", H);
            require("model.embed_tokens.weight", H, kGenerationVocabSize);
            require("lm_head.weight", H, kGenerationVocabSize);
            if (fixed.adapts(model.get("lm_head.weight")))
                throw std::runtime_error("semantic head slicing requires an unadapted LM head");
            persistent = context();
            std::mt19937 rng(options.seed);
            for (int layer = 0; layer < L; ++layer) {
                const auto base = "model.layers." + std::to_string(layer) + ".";
                for (const auto &suffix :
                     {"self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj",
                      "self_attn.o_proj", "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj"}) {
                    const auto stem = base + suffix;
                    auto *w = model.get(stem + ".weight");
                    if(options.dora) {
                        auto values=fixed.row_norm_squared(w);
                        auto *norm=ggml_new_tensor_1d(persistent.get(),GGML_TYPE_F32,w->ne[1]);
                        ggml_set_input(norm);base_norms[stem]=norm;base_norm_values[stem]=std::move(values);
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
                        } else if(side==2) {
                            const auto &norm=base_norm_values.at(stem);
                            for(size_t i=0;i<norm.size();++i)p.values[i]=std::sqrt(norm[i]+1e-12F);
                        }
                        p.gradient.resize(p.values.size());
                        p.m.resize(p.values.size());
                        p.v.resize(p.values.size());
                        parameter_index[p.name] = parameters.size();
                        parameters.push_back(std::move(p));
                    }
                }
            }
            for (int layer = 0; layer <= L; ++layer) {
                auto *boundary = tensor(H, n);
                ggml_set_param(boundary);
                boundaries.push_back(boundary);
            }
            mask = tensor(n, n);
            incoming = tensor(H, n);
            outgoing = tensor(H, n);
            prediction = tensor(kCodecSize + 1, targets);
            target = tensor(kCodecSize + 1, targets);
            if (options.kl_weight > 0) {
                teacher = tensor(kCodecSize + 1, targets);
                kl_scale = tensor(1, 1);
                teacher_entropy = tensor(1, 1);
            }
            positions = ggml_new_tensor_1d(persistent.get(), GGML_TYPE_I32, n);
            token_ids = ggml_new_tensor_1d(persistent.get(), GGML_TYPE_I32, n);
            ggml_set_input(positions);
            ggml_set_input(token_ids);
            buffer = ggml_backend_alloc_ctx_tensors(persistent.get(), model.backend());
            if (!buffer)
                throw std::runtime_error("cannot allocate training state");
            spans = detail::tensor_spans(parameters, size_t(1) << 16);
            for (auto &p : parameters)
                set(p.tensor, p.values);
            for(const auto &entry:base_norms)set(entry.second,base_norm_values.at(entry.first));
            build_front();
            build_tail();
            for (int layer = 0; layer < L; ++layer)
                build_block(layer);
            std::map<std::string, ggml_gallocr_t> shapes;
            for (auto *graph : graphs()) {
                std::string shape;
                for (int i = 0; i < ggml_graph_n_nodes(graph); ++i) {
                    auto *node = ggml_graph_node(graph, i);
                    if (node->op == GGML_OP_MUL_MAT)
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
                         "[yue2:ar-train] %u frames, %u prefix tokens, rank %d, 28 "
                         "layers, %zu workspace "
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
    ggml_tensor *linear(ggml_context *ctx, ggml_tensor *x, const std::string &name,
                        bool train = true) {
        auto *output = fixed.linear(ctx, model.get(name + ".weight"), x, true);
        auto found = parameter_index.find(name + ".lora_A");
        if (train && found != parameter_index.end()) {
            auto *a = parameters[found->second].tensor;
            auto *b = parameters[found->second + 1].tensor;
            output = ggml_add(ctx, output, ggml_mul_mat(ctx, b, ggml_mul_mat(ctx, a, x)));
            if(options.dora) {
                auto *projection=fixed.linear(ctx,model.get(name+".weight"),a,true);
                auto *denom=detail::dora_norm_from_projection(ctx,projection,a,b,base_norms.at(name));
                auto *m=parameters[found->second+2].tensor;
                output=ggml_mul(ctx,output,ggml_div(ctx,ggml_reshape_1d(ctx,m,m->ne[0]),denom));
            }
        }
        return output;
    }
    int parameters_per_layer() const { return options.dora?21:14; }
    ggml_tensor *block(ggml_context *ctx, ggml_tensor *x, int layer, bool adapt=true) {
        const auto base = "model.layers." + std::to_string(layer) + ".";
        auto *normalized = norm(ctx, x, model.get(base + "input_layernorm.weight"));
        const auto attn = base + "self_attn.";
        auto *q = ggml_reshape_3d(ctx, linear(ctx, normalized, attn + "q_proj",adapt), D, Q, n);
        auto *k = ggml_reshape_3d(ctx, linear(ctx, normalized, attn + "k_proj",adapt), D, K, n);
        auto *v = ggml_reshape_3d(ctx, linear(ctx, normalized, attn + "v_proj",adapt), D, K, n);
        q = norm(ctx, q, model.get(attn + "q_norm.weight"));
        k = norm(ctx, k, model.get(attn + "k_norm.weight"));
        q = ggml_rope_ext(ctx, q, positions, nullptr, D, GGML_ROPE_TYPE_NEOX, 0, 1e6F, 1, 0, 1, 0,
                          0);
        k = ggml_rope_ext(ctx, k, positions, nullptr, D, GGML_ROPE_TYPE_NEOX, 0, 1e6F, 1, 0, 1, 0,
                          0);
        q = ggml_permute(ctx, q, 0, 2, 1, 3);
        k = ggml_cont(ctx, ggml_permute(ctx, k, 0, 2, 1, 3));
        v = ggml_cont(ctx, ggml_permute(ctx, v, 0, 2, 1, 3));
        auto *scores =
            ggml_soft_max_ext(ctx, ggml_mul_mat(ctx, k, q), mask, 1 / std::sqrt(float(D)), 0);
        auto *attended = ggml_mul_mat(ctx, ggml_cont(ctx, ggml_transpose(ctx, v)), scores);
        attended =
            ggml_reshape_2d(ctx, ggml_cont(ctx, ggml_permute(ctx, attended, 0, 2, 1, 3)), H, n);
        x = ggml_add(ctx, x, linear(ctx, attended, attn + "o_proj",adapt));
        normalized = norm(ctx, x, model.get(base + "post_attention_layernorm.weight"));
        auto *gate = linear(ctx, normalized, base + "mlp.gate_proj",adapt);
        auto *up = linear(ctx, normalized, base + "mlp.up_proj",adapt);
        return ggml_add(ctx, x,
                        linear(ctx, ggml_swiglu_split(ctx, gate, up), base + "mlp.down_proj",adapt));
    }
    void build_front() {
        front.ctx = context();
        auto *ctx = front.ctx.get();
        auto *hidden = ggml_get_rows(ctx, model.get("model.embed_tokens.weight"), token_ids);
        front.forward = ggml_new_graph_custom(ctx, C, false);
        ggml_build_forward_expand(front.forward, ggml_cpy(ctx, hidden, boundaries[0]));
    }
    void build_tail() {
        tail.ctx = context(true);
        auto *ctx = tail.ctx.get();
        auto *hidden = norm(ctx, boundaries.back(), model.get("model.norm.weight"));
        auto *selected = ggml_cont(
            ctx, ggml_view_2d(ctx, hidden, H, targets, hidden->nb[1], (ar - 1) * hidden->nb[1]));
        auto *head = model.get("lm_head.weight");
        auto *semantic_head =
            ggml_view_2d(ctx, head, H, kCodecSize + 1, head->nb[1], kMusicEndToken * head->nb[1]);
        auto *logits = ggml_mul_mat(ctx, f32(ctx, semantic_head), selected);
        tail.task_loss = ggml_cross_entropy_loss(ctx, logits, target);
        ggml_set_output(tail.task_loss);
        tail.loss = tail.task_loss;
        if (options.kl_weight > 0) {
            tail.kl_loss =
                ggml_sub(ctx, ggml_cross_entropy_loss(ctx, logits, teacher), teacher_entropy);
            ggml_set_output(tail.kl_loss);
            tail.loss = ggml_add(ctx, tail.task_loss, ggml_mul(ctx, kl_scale, tail.kl_loss));
        }
        ggml_set_loss(tail.loss);
        ggml_set_output(tail.loss);
        tail.backward = ggml_new_graph_custom(ctx, C, true);
        ggml_build_forward_expand(tail.backward, ggml_cpy(ctx, logits, prediction));
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
        if(options.dora || options.online_teacher) {
            graph.teacher_ctx=context();ctx=graph.teacher_ctx.get();
            auto *common=block(ctx,boundaries[layer],layer,false);
            graph.teacher_forward=ggml_new_graph_custom(ctx,C,false);
            ggml_build_forward_expand(graph.teacher_forward,ggml_cpy(ctx,common,boundaries[layer+1]));
        }
    }
    std::vector<ggml_cgraph *> graphs() {
        std::vector<ggml_cgraph *> result{front.forward, tail.backward};
        for (auto &b : blocks) {
            result.push_back(b.forward);
            result.push_back(b.backward);
            if(b.teacher_forward)result.push_back(b.teacher_forward);
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
    void inputs(const ArTrainingSample &sample, bool regularize = false) {
        if (sample.prefix.empty() || sample.prefix.size() > ar || sample.codes.size() != frames ||
            sample.prefix.back() != kMusicStartToken)
            throw std::invalid_argument("AR sample shape or MUSIC_START prefix mismatch");
        last_actual_ar = static_cast<uint32_t>(sample.prefix.size());
        const auto pad = ar - last_actual_ar;
        std::vector<int32_t> ids(pad, kEodToken);
        ids.insert(ids.end(), sample.prefix.begin(), sample.prefix.end());
        std::vector<float> labels(size_t(kCodecSize + 1) * targets, 0);
        for (auto token : ids)
            if (token < 0 || token >= kGenerationVocabSize)
                throw std::invalid_argument("AR prefix token out of range");
        for (uint32_t i = 0; i < frames; ++i) {
            auto code = sample.codes[i];
            if (code < 0 || code >= kCodecSize)
                throw std::invalid_argument("AR codec target out of range");
            labels[size_t(i) * (kCodecSize + 1) + code + 1] = 1;
            if (predict_end || i + 1 < frames)
                ids.push_back(kCodecOffset + code);
        }
        if (predict_end)
            labels[size_t(frames) * (kCodecSize + 1)] = 1;
        if (ids.size() != n)
            throw std::runtime_error("AR teacher forcing shift mismatch");
        ggml_backend_tensor_set(token_ids, ids.data(), 0, ids.size() * sizeof(ids[0]));
        std::vector<int32_t> pos(n);
        std::vector<float> attention_mask(size_t(n) * n, 0);
        for (uint32_t i = 0; i < n; ++i) {
            pos[i] = i >= pad ? i - pad : 0;
            for (uint32_t j = 0; j < n; ++j)
                if (j > i || (i >= pad && j < pad))
                    attention_mask[size_t(i) * n + j] = -std::numeric_limits<float>::infinity();
        }
        ggml_backend_tensor_set(positions, pos.data(), 0, pos.size() * sizeof(pos[0]));
        set(mask, attention_mask);
        set(target, labels);
        last_kl_scale = regularize ? options.kl_weight : 0;
        if (teacher) {
            if (last_kl_scale > 0) {
                if (sample.teacher_probabilities.size() != labels.size() ||
                    !std::isfinite(sample.teacher_entropy))
                    throw std::runtime_error("missing/invalid frozen teacher distribution");
                // Prepared distributions were checked completely once.
                if (!sample.teacher_validated)
                    check_teacher(sample);
                set(teacher, sample.teacher_probabilities);
                set(teacher_entropy, {sample.teacher_entropy});
            } else {
                set(teacher, labels);
                set(teacher_entropy, {0});
            }
            set(kl_scale, {last_kl_scale});
        }
    }
    void check_teacher(const ArTrainingSample &sample) const {
        if (sample.teacher_probabilities.size() != size_t(kCodecSize + 1) * targets ||
            !std::isfinite(sample.teacher_entropy))
            throw std::runtime_error("missing/invalid frozen teacher distribution");
        const auto &probabilities = sample.teacher_probabilities;
        // Rows are independent; each row is summed serially as before. The
        // first failing check in the former serial order is reported.
        std::vector<char> status(targets, 0);
        detail::parallel_tasks(targets, options.threads, [&](size_t row) {
            const auto *data = probabilities.data() + row * (kCodecSize + 1);
            if (!detail::all_finite(data, 0, kCodecSize + 1)) {
                status[row] = 1;
                return;
            }
            double sum = 0;
            for (int col = 0; col <= kCodecSize; ++col) {
                if (data[col] < 0) {
                    status[row] = 2;
                    return;
                }
                sum += data[col];
            }
            if (std::abs(sum - 1) > 1e-4)
                status[row] = 3;
        });
        if (std::find(status.begin(), status.end(), 1) != status.end())
            throw std::runtime_error("nonfinite teacher probabilities");
        const auto failed = std::find_if(status.begin(), status.end(), [](char s) { return s != 0; });
        if (failed != status.end())
            throw std::runtime_error(*failed == 2 ? "negative teacher probability"
                                                  : "unnormalized teacher row");
    }
    void validate_teacher(ArTrainingSample &sample) const {
        sample.teacher_validated = false;
        check_teacher(sample);
        sample.teacher_validated = true;
    }
    float forward(bool common_teacher=false) {
        compute(front.forward);
        for (auto &graph : blocks)
            compute(common_teacher && graph.teacher_forward ? graph.teacher_forward : graph.forward);
        compute(tail.backward, true);
        const auto loss = get(tail.loss)[0];
        if (!std::isfinite(loss))
            throw std::runtime_error("nonfinite training loss");
        return loss;
    }
    ArTrainingResult step(const ArTrainingSample &sample, bool update) {
        const auto inputs_start = Clock::now();
        inputs(sample, true);
        const double inputs_seconds = since(inputs_start);
        const float loss = forward();
        double optimizer_seconds = 0;
        const float cross_entropy = get(tail.task_loss)[0];
        const float teacher_kl = tail.kl_loss ? get(tail.kl_loss)[0] : 0;
        double squared = 0;
        std::future<void> scan;
        for (int layer = L - 1; layer >= 0; --layer) {
            auto &graph = blocks[layer];
            compute(graph.backward, true);
            const auto download_start = Clock::now();
            const int first = layer * parameters_per_layer(), last = first + parameters_per_layer();
            for (int i = first; i < last; ++i)
                download(ggml_graph_get_grad(graph.backward, parameters[i].tensor), parameters[i].gradient);
            // Finite checks and the norm reduction overlap the next layer's
            // backward pass. Each scan waits for the previous one, so the
            // global sum keeps exactly the former serial order.
            scan = std::async(std::launch::async,
                              [this, first, last, &squared, previous = std::move(scan)]() mutable {
                                  if (previous.valid())
                                      previous.get();
                                  for (int i = first; i < last; ++i) {
                                      const auto &gradient = parameters[i].gradient;
                                      finite(gradient, "adapter gradient");
                                      for (auto value : gradient)
                                          squared += double(value) * value;
                                  }
                              });
            optimizer_seconds += since(download_start);
            ggml_backend_tensor_copy(outgoing, incoming);
        }
        const auto scan_start = Clock::now();
        scan.get();
        optimizer_seconds += since(scan_start);
        const auto gradient_norm = std::sqrt(squared);
        if (!std::isfinite(gradient_norm))
            throw std::runtime_error("nonfinite global gradient norm");
        if (update) {
            const auto update_start = Clock::now();
            ++optimizer_step;
            const double scale = options.gradient_clip > 0 && gradient_norm > options.gradient_clip
                                     ? options.gradient_clip / gradient_norm
                                     : 1;
            const double correction1 = 1 - std::pow(0.9, double(optimizer_step));
            const double correction2 = 1 - std::pow(0.999, double(optimizer_step));
            // Elementwise, so each span is independent of thread scheduling.
            std::vector<char> ok(spans.size(), 1);
            detail::parallel_tasks(spans.size(), options.threads, [&](size_t task) {
                const auto &span = spans[task];
                auto &p = parameters[span.tensor];
                ok[task] = detail::adamw_span(p.values.data(), p.gradient.data(), p.m.data(),
                                              p.v.data(), span.begin, span.end, scale, correction1,
                                              correction2, options.learning_rate,
                                              options.weight_decay);
            });
            if (std::find(ok.begin(), ok.end(), 0) != ok.end())
                throw std::runtime_error("nonfinite updated adapter");
            for (auto &p : parameters)
                set(p.tensor, p.values);
            optimizer_seconds += since(update_start);
        }
        return {loss, gradient_norm, optimizer_step, cross_entropy, teacher_kl, inputs_seconds,
                optimizer_seconds};
    }
    void prepare_teacher(ArTrainingSample &sample) {
        if (optimizer_step != 0)
            throw std::runtime_error("prepare teacher before resume/update");
        for (size_t i = 1; i < parameters.size(); i += options.dora?3:2)
            for (auto value : parameters[i].values)
                if (value != 0)
                    throw std::runtime_error("teacher requires zero learned B factors");
        distribution(sample);
    }
    void common_teacher(ArTrainingSample &sample) {
        if (!options.online_teacher)
            throw std::runtime_error("online teacher requires ArTrainingOptions::online_teacher");
        distribution(sample);
    }
    // With zero B factors the adapted block adds exactly zero, so the common
    // graph reproduces prepare_teacher's distribution bit for bit.
    void distribution(ArTrainingSample &sample) {
        inputs(sample, false);
        forward(true);
        auto probabilities = get(prediction);
        double entropy = 0;
        for (uint32_t row = 0; row < targets; ++row) {
            auto *data = probabilities.data() + size_t(row) * (kCodecSize + 1);
            const auto largest = *std::max_element(data, data + kCodecSize + 1);
            double sum = 0;
            for (int c = 0; c <= kCodecSize; ++c) {
                data[c] = static_cast<float>(std::exp(double(data[c] - largest)));
                sum += data[c];
            }
            for (int c = 0; c <= kCodecSize; ++c) {
                data[c] = static_cast<float>(data[c] / sum);
                if (data[c] > 0)
                    entropy -= double(data[c]) * std::log(double(data[c]));
            }
        }
        sample.teacher_probabilities = std::move(probabilities);
        sample.teacher_entropy = static_cast<float>(entropy / targets);
        validate_teacher(sample);
    }
    void metadata(gguf_context *file, bool state = false) const {
        gguf_set_val_str(file, "general.architecture", state ? "yue2_ar_training" : "yue2_lora");
        gguf_set_val_str(file, "yue2.component",
                         state ? "ar-training-state" : "generation-adapter");
        gguf_set_val_str(file, "yue2.adapter.type", options.dora?"dora":"lora");
        if(options.dora) {
            gguf_set_val_str(file,"yue2.adapter.dora.norm_gradient","full");
            gguf_set_val_u32(file,"yue2.adapter.dora.target_count",L*7);
            gguf_set_val_str(file,"yue2.adapter.dora.reference_prefix",reference_id.c_str());
        }
        gguf_set_val_u32(file, "yue2.adapter.rank", options.rank);
        gguf_set_val_f32(file, "yue2.adapter.alpha", float(options.rank));
        gguf_set_val_u32(file, "yue2.adapter.target_count", L * 7);
        if (auto sha = model.optional_string("yue2.checkpoint.sha256"))
            gguf_set_val_str(file, "yue2.adapter.base_sha256", sha->c_str());
        gguf_set_val_str(file, "yue2.training.identity", options.run_identity.c_str());
        gguf_set_val_str(file, "yue2.training.fixed_adapters", fixed_identity().c_str());
        gguf_set_val_u64(file, "yue2.training.step", optimizer_step);
        gguf_set_val_str(file, "yue2.training.step_text", std::to_string(optimizer_step).c_str());
        gguf_set_val_u32(file, "yue2.training.frames", frames);
        gguf_set_val_u32(file, "yue2.training.ar_slots", ar);
        gguf_set_val_u32(file, "yue2.training.predict_end", predict_end ? 1 : 0);
        gguf_set_val_str(file, "yue2.training.objective",
                         "semantic-only cross-entropy; no planning loss");
        gguf_set_val_str(file, "yue2.training.projection_precision", "f32-compute-f16-storage");
        gguf_set_val_str(file, "yue2.training.prefix_policy",
                         "left-padding-masked-real-position-zero");
        gguf_set_val_f32(file, "yue2.training.learning_rate", options.learning_rate);
        gguf_set_val_f32(file, "yue2.training.weight_decay", options.weight_decay);
        gguf_set_val_f32(file, "yue2.training.gradient_clip", options.gradient_clip);
        gguf_set_val_f32(file, "yue2.training.kl_weight", options.kl_weight);
        gguf_set_val_str(file, "yue2.training.kl_weight_text",
                         std::to_string(options.kl_weight).c_str());
        gguf_set_val_str(file, "yue2.training.composition",
                         "AR delta stacked after fixed adapters; keep fixed "
                         "adapters and use base planner separately");
    }
    std::string fixed_identity() const {
        std::string result;
        for (const auto &spec : options.fixed_adapters)
            result += std::filesystem::absolute(spec.path).lexically_normal().string() + "=" +
                      std::to_string(spec.strength) + "\n";
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
        const auto gains=options.dora && !state && !diagnostics ? export_row_gains()
            : std::map<std::string,std::vector<float>>{};
        for(const auto &entry:gains)add(entry.first+".dora_scale",entry.second,entry.second.size());
        if (diagnostics) {
            gguf_set_val_u32(file, "yue2.training.actual_ar_tokens", last_actual_ar);
            auto snapshot = [&](const std::string &name, ggml_tensor *t) {
                extra.push_back(get(t));
                add(name, extra.back(), t->ne[0], t->ne[1], t->ne[2]);
            };
            std::vector<int32_t> ids(n);
            ggml_backend_tensor_get(token_ids, ids.data(), 0, ids.size() * sizeof(ids[0]));
            extra.emplace_back();
            for (auto id : ids)
                extra.back().push_back(static_cast<float>(id));
            add("token_ids", extra.back(), n);
            snapshot("target", target);
            snapshot("logits", prediction);
            if (teacher) {
                snapshot("teacher_probabilities", teacher);
                gguf_set_val_f32(file, "yue2.training.teacher_entropy", get(teacher_entropy)[0]);
                gguf_set_val_f32(file, "yue2.training.active_kl_weight", last_kl_scale);
            }
        }
        if (state)
            gguf_set_val_str(file, "yue2.training.rng", rng.c_str());
        write_file(file, path);
    }
    std::string load(const std::string &path) {
        auto saved = detail::load_gguf_raw(path.c_str(), "cpu", 1);
        if (saved.string("general.architecture") != "yue2_ar_training" ||
            saved.string("yue2.adapter.type") != (options.dora?"dora":"lora") ||
            (options.dora && (saved.optional_string("yue2.adapter.dora.norm_gradient").value_or("")!="full" ||
             saved.optional_string("yue2.adapter.dora.reference_prefix").value_or("")!=reference_id)) ||
            saved.optional_string("yue2.training.kl_weight_text").value_or("0.000000") !=
                std::to_string(options.kl_weight) ||
            saved.string("yue2.training.projection_precision") != "f32-compute-f16-storage" ||
            saved.u32("yue2.adapter.rank") != static_cast<uint32_t>(options.rank) ||
            saved.u32("yue2.training.frames") != frames ||
            saved.u32("yue2.training.ar_slots") != ar ||
            saved.u32("yue2.training.predict_end") != uint32_t(predict_end) ||
            saved.string("yue2.training.identity") != options.run_identity ||
            saved.f32("yue2.training.learning_rate") != options.learning_rate ||
            saved.string("yue2.training.fixed_adapters") != fixed_identity() ||
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
    std::map<std::string,std::vector<float>> export_row_gains() const {
        std::map<std::string,std::vector<float>> result;
        // Export runs between updates. Reuse the largest existing scratch
        // buffer instead of allocating another projection workspace while all
        // training buffers are resident. gallocr replans each subsequent graph
        // on alloc_graph, just as it already does for the shared layer graphs.
        auto allocator=*std::max_element(workspaces.begin(),workspaces.end(),[](auto a,auto b){
            return ggml_gallocr_get_buffer_size(a,0)<ggml_gallocr_get_buffer_size(b,0);
        });
        for(const auto &entry:base_norms) {
            auto ctx=context();const auto at=parameter_index.at(entry.first+".lora_A");
            auto *a=parameters[at].tensor,*b=parameters[at+1].tensor;
            auto *projection=fixed.linear(ctx.get(),model.get(entry.first+".weight"),a,true);
            auto *denom=detail::dora_norm_from_projection(ctx.get(),projection,a,b,entry.second);
            ggml_set_output(denom);auto *graph=ggml_new_graph_custom(ctx.get(),C,false);ggml_build_forward_expand(graph,denom);
            for(int i=0;i<ggml_graph_n_nodes(graph);++i) {
                auto *node=ggml_graph_node(graph,i);if(node->op==GGML_OP_MUL_MAT)ggml_mul_mat_set_prec(node,GGML_PREC_F32);
            }
            if(!allocator||!ggml_gallocr_alloc_graph(allocator,graph)||ggml_backend_graph_compute(model.backend(),graph)!=GGML_STATUS_SUCCESS)
                throw std::runtime_error("AR DoRA row-gain export failed");
            auto values=get(denom);const auto &m=parameters[at+2].values;const auto &base=base_norm_values.at(entry.first);
            for(size_t i=0;i<values.size();++i) {
                if(!std::isfinite(values[i])||values[i]*values[i]<1e-8F*base[i])throw std::runtime_error("AR DoRA direction nearly cancels its frozen base");
                values[i]=m[i]/values[i];
            }
            finite(values,"AR DoRA row gain");result[entry.first]=std::move(values);
        }
        return result;
    }
    ArTrainingOptions options;
    uint32_t frames, ar, n, targets, last_actual_ar = 0;
    bool predict_end;
    detail::GgufModel model;
    detail::LoraStack fixed;
    Context persistent{nullptr, ggml_free};
    ggml_backend_buffer_t buffer = nullptr;
    std::vector<ggml_gallocr_t> workspaces;
    std::map<ggml_cgraph *, ggml_gallocr_t> graph_allocators;
    std::vector<Parameter> parameters;
    std::vector<detail::TensorSpan> spans;
    std::map<std::string, size_t> parameter_index;
    std::map<std::string,ggml_tensor *> base_norms;
    std::map<std::string,std::vector<float>> base_norm_values;
    std::string reference_id;
    std::vector<ggml_tensor *> boundaries;
    ggml_tensor *target = nullptr, *mask = nullptr, *positions = nullptr, *token_ids = nullptr;
    ggml_tensor *incoming = nullptr, *outgoing = nullptr, *prediction = nullptr;
    ggml_tensor *teacher = nullptr, *teacher_entropy = nullptr, *kl_scale = nullptr;
    float last_kl_scale = 0;
    Graphs front, tail;
    std::vector<Graphs> blocks;
    uint64_t optimizer_step = 0;
};

ArTrainer::ArTrainer(const std::string &model, uint32_t prefix, uint32_t frames, bool predict_end,
                     const ArTrainingOptions &options)
    : impl_(std::make_unique<Impl>(model, prefix, frames, predict_end, options)) {}
ArTrainer::~ArTrainer() = default;
ArTrainingResult ArTrainer::step(const ArTrainingSample &s, bool update) {
    return impl_->step(s, update);
}
float ArTrainer::evaluate(const ArTrainingSample &s) {
    impl_->inputs(s);
    return impl_->forward();
}
void ArTrainer::prepare_teacher(ArTrainingSample &s) { impl_->prepare_teacher(s); }
void ArTrainer::teacher(ArTrainingSample &s) { impl_->common_teacher(s); }
void ArTrainer::validate_teacher(ArTrainingSample &s) { impl_->validate_teacher(s); }
void ArTrainer::save_adapter(const std::string &path) const { impl_->write(path, false); }
void ArTrainer::save_state(const std::string &path, const std::string &rng) const {
    impl_->write(path, true, rng);
}
std::string ArTrainer::load_state(const std::string &path) { return impl_->load(path); }
void ArTrainer::save_diagnostics(const std::string &path) const {
    impl_->write(path, false, "", true);
}
} // namespace yue2
