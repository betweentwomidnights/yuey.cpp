// Model-free NAR-block gradient feasibility probe. No musical-quality claim.
// Exercises GQA, frozen AR K/V, NeoX RoPE, RMSNorm and split SwiGLU with
// functional LoRAs and frozen F32/F16/Q4_K projections on the chosen backend.
#include "gguf_model.h"
#include "ggml-alloc.h"

#include <algorithm>
#include <cmath>
#include <iostream>
#include <random>

namespace {
constexpr int hidden = 256, intermediate = 512, heads = 4, kvheads = 2;
constexpr int dim = 64, frames = 6, prefix = 3, rank = 2;
constexpr std::size_t capacity = 8192;
struct Parameter { ggml_tensor * tensor; std::vector<float> values; };

class Probe {
public:
    Probe(const std::string & device, ggml_type type) : type_(type) {
        backend_ = yue2::detail::make_backend(device.c_str(), 4);
        persistent_ = ggml_init({ggml_tensor_overhead()*128, nullptr, true});
        context_ = ggml_init({ggml_tensor_overhead()*capacity + ggml_graph_overhead_custom(capacity, true), nullptr, true});
        if (!persistent_ || !context_) throw std::runtime_error("context allocation failed");
        auto * x = input("block_input", hidden, frames, true);
        auto * positions = ggml_new_tensor_1d(persistent_, GGML_TYPE_I32, frames);
        ggml_set_input(positions);
        auto * ar_key = input("frozen_ar_key", dim, prefix*kvheads, false);
        auto * ar_value = input("frozen_ar_value", dim, prefix*kvheads, false);
        auto * norm = input("norm", hidden, 1, false, true);
        auto * qnorm = input("qnorm", dim, 1, false, true);
        auto normalize = [&](ggml_tensor * a, ggml_tensor * weight) {
            return ggml_mul(context_, ggml_rms_norm(context_, a, 1e-6F), weight);
        };
        auto * n = normalize(x, norm);
        auto * q = linear("q", n, hidden);
        auto * k = linear("k", n, dim*kvheads);
        auto * v = linear("v", n, dim*kvheads);
        q = normalize(ggml_reshape_3d(context_, q, dim, heads, frames), qnorm);
        k = normalize(ggml_reshape_3d(context_, k, dim, kvheads, frames), qnorm);
        q = ggml_rope_ext(context_, q, positions, nullptr, dim, GGML_ROPE_TYPE_NEOX, 0,
                          1e6F, 1, 0, 1, 0, 0);
        k = ggml_rope_ext(context_, k, positions, nullptr, dim, GGML_ROPE_TYPE_NEOX, 0,
                          1e6F, 1, 0, 1, 0, 0);
        q = ggml_permute(context_, q, 0, 2, 1, 3);
        k = ggml_cont(context_, ggml_permute(context_, k, 0, 2, 1, 3));
        v = ggml_cont(context_, ggml_permute(context_,
            ggml_reshape_3d(context_, v, dim, kvheads, frames), 0, 2, 1, 3));
        k = ggml_concat(context_, ggml_reshape_3d(context_, ar_key, dim, prefix, kvheads), k, 1);
        v = ggml_concat(context_, ggml_reshape_3d(context_, ar_value, dim, prefix, kvheads), v, 1);
        auto * scores = ggml_soft_max_ext(context_, ggml_mul_mat(context_, k, q), nullptr,
                                         1/std::sqrt(float(dim)), 0);
        auto * attention = ggml_mul_mat(context_, ggml_cont(context_, ggml_transpose(context_, v)), scores);
        attention = ggml_reshape_2d(context_, ggml_cont(context_,
            ggml_permute(context_, attention, 0, 2, 1, 3)), hidden, frames);
        auto * residual = ggml_add(context_, x, linear("o", attention, hidden));
        n = normalize(residual, norm);
        auto * gate = linear("gate", n, intermediate);
        auto * up = linear("up", n, intermediate);
        auto * output = ggml_add(context_, residual, linear("down", ggml_swiglu_split(context_, gate, up), hidden));
        auto * target = input("target", hidden, frames, false);
        auto * difference = ggml_sub(context_, output, target);
        loss_ = ggml_scale(context_, ggml_sum(context_, ggml_sqr(context_, difference)), 1.0F/(hidden*frames));
        ggml_set_loss(loss_);
        ggml_set_output(loss_);
        graph_ = ggml_new_graph_custom(context_, capacity, true);
        ggml_build_forward_expand(graph_, loss_);
        // Finite differences need stable accumulation; CUDA's default F32
        // projection policy may otherwise round inputs through tensor cores.
        for (int i = 0; i < ggml_graph_n_nodes(graph_); ++i) {
            auto * node = ggml_graph_node(graph_, i);
            if (node->op == GGML_OP_MUL_MAT) ggml_mul_mat_set_prec(node, GGML_PREC_F32);
        }
        ggml_build_backward_expand(context_, graph_, nullptr);
        for (const auto & p : parameters_) {
            auto * gradient = ggml_graph_get_grad(graph_, p.tensor);
            if (!gradient) throw std::runtime_error("missing parameter gradient");
            ggml_set_output(gradient);
        }
        buffer_ = ggml_backend_alloc_ctx_tensors(persistent_, backend_);
        if (!buffer_) throw std::runtime_error("persistent allocation failed");
        allocator_ = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend_));
        if (!ggml_gallocr_alloc_graph(allocator_, graph_)) throw std::runtime_error("graph allocation failed");
        for (int i = 0; i < ggml_graph_n_nodes(graph_); ++i) {
            auto * node = ggml_graph_node(graph_, i);
            if (node->op != GGML_OP_NONE && !ggml_backend_supports_op(backend_, node)) {
                throw std::runtime_error(std::string("backend does not support ") + ggml_op_name(node->op));
            }
        }
        for (const auto & p : inputs_) set(p.tensor, p.values);
        for (const auto & p : parameters_) set(p.tensor, p.values);
        for (const auto & p : weights_) {
            std::vector<std::uint8_t> bytes(ggml_nbytes(p.tensor));
            ggml_quantize_chunk(p.tensor->type, p.values.data(), bytes.data(), 0, p.tensor->ne[1], p.tensor->ne[0], nullptr);
            ggml_backend_tensor_set(p.tensor, bytes.data(), 0, bytes.size());
        }
        std::vector<std::int32_t> pos(frames);
        for (int i = 0; i < frames; ++i) pos[i] = prefix+i;
        ggml_backend_tensor_set(positions, pos.data(), 0, pos.size()*sizeof(pos[0]));
    }
    ~Probe() {
        if (allocator_) ggml_gallocr_free(allocator_);
        if (buffer_) ggml_backend_buffer_free(buffer_);
        if (context_) ggml_free(context_);
        if (persistent_) ggml_free(persistent_);
        if (backend_) ggml_backend_free(backend_);
    }
    void check() {
        const auto initial = compute();
        std::vector<std::vector<float>> gradients;
        for (const auto & p : parameters_) {
            auto * gradient = ggml_graph_get_grad(graph_, p.tensor);
            if (!gradient) throw std::runtime_error("missing gradient");
            std::vector<float> values(p.values.size());
            ggml_backend_tensor_get(gradient, values.data(), 0, values.size()*sizeof(float));
            double squared = 0;
            for (auto value : values) {
                if (!std::isfinite(value)) throw std::runtime_error("nonfinite gradient");
                squared += double(value)*value;
            }
            if (squared < 1e-20) throw std::runtime_error("zero gradient");
            gradients.push_back(std::move(values));
        }
        // Check the largest-magnitude entry in every gradient, including the
        // block boundary needed for activation recomputation through 28 layers.
        double worst = 0;
        int mismatches = 0;
        for (std::size_t i = 0; i < parameters_.size(); ++i) {
            auto & p = parameters_[i];
            const auto & grad = gradients[i];
            const auto at = std::max_element(grad.begin(), grad.end(), [](float a, float b) { return std::abs(a)<std::abs(b); }) - grad.begin();
            // Small differences resolve CPU F32 tightly. CUDA's projection
            // kernels and quantized activations require a wider finite-
            // difference stencil; use the standard GGML grad-test step there.
            const float original = p.values[at];
            const float epsilon = ggml_backend_dev_type(ggml_backend_get_device(backend_)) == GGML_BACKEND_DEVICE_TYPE_CPU &&
                                  type_ == GGML_TYPE_F32 ? 0.01F : 0.1F;
            p.values[at] = original+epsilon; set(p.tensor, p.values); const auto plus = compute();
            p.values[at] = original-epsilon; set(p.tensor, p.values); const auto minus = compute();
            p.values[at] = original+2*epsilon; set(p.tensor, p.values); const auto plus2 = compute();
            p.values[at] = original-2*epsilon; set(p.tensor, p.values); const auto minus2 = compute();
            p.values[at] = original; set(p.tensor, p.values);
            const float numerical = (8*(plus-minus) - (plus2-minus2))/(12*epsilon);
            const double relative = std::abs(numerical-grad[at])/std::max(0.00002F, std::abs(grad[at]));
            worst = std::max(worst, relative);
            std::cout << ggml_get_name(p.tensor) << " analytical=" << grad[at]
                      << " numerical=" << numerical << " relative=" << relative << '\n';
            if (std::abs(numerical-grad[at]) > 0.00003F + 0.03F*std::abs(grad[at]))
                ++mismatches;
        }
        if (mismatches) throw std::runtime_error("finite difference gradient mismatches: " + std::to_string(mismatches));
        // Hold the block input fixed; update only the fourteen adapter factors.
        for (std::size_t i = 1; i < parameters_.size(); ++i) {
            auto & p = parameters_[i];
            for (std::size_t j = 0; j < p.values.size(); ++j) p.values[j] -= 0.001F*gradients[i][j];
            set(p.tensor, p.values);
        }
        const auto after = compute();
        if (!(after < initial)) throw std::runtime_error("adapter update did not lower loss");
        std::cout << "PASS type=" << ggml_type_name(type_) << " params=" << parameters_.size()-1
                  << " loss=" << initial << " after=" << after << " worst_relative=" << worst << '\n';
    }
private:
    void set(ggml_tensor * t, const std::vector<float> & values) {
        ggml_backend_tensor_set(t, values.data(), 0, values.size()*sizeof(float));
    }
    std::vector<float> random(std::size_t n, float scale) {
        std::normal_distribution<float> dist(0, scale);
        std::vector<float> values(n);
        for (auto & v : values) v = dist(rng_);
        return values;
    }
    ggml_tensor * input(const char * name, int in, int out, bool parameter, bool ones=false) {
        auto * tensor = ggml_new_tensor_2d(persistent_, GGML_TYPE_F32, in, out);
        ggml_set_name(tensor, name);
        ggml_set_input(tensor);
        if (parameter) ggml_set_param(tensor);
        Parameter p{tensor, ones ? std::vector<float>(in*out, 1) : random(in*out, 0.2F)};
        (parameter ? parameters_ : inputs_).push_back(std::move(p));
        return tensor;
    }
    ggml_tensor * linear(const char * name, ggml_tensor * x, int out) {
        const int in = static_cast<int>(x->ne[0]);
        auto * w = ggml_new_tensor_2d(persistent_, type_, in, out);
        ggml_set_name(w, (std::string(name)+".frozen").c_str());
        weights_.push_back({w, random(in*out, 0.2F/std::sqrt(float(in)))});
        auto * a = input((std::string(name)+".lora_A").c_str(), in, rank, true);
        auto * b = input((std::string(name)+".lora_B").c_str(), rank, out, true);
        return ggml_add(context_, ggml_mul_mat(context_, w, x),
            ggml_mul_mat(context_, b, ggml_mul_mat(context_, a, x)));
    }
    float compute() {
        ggml_graph_reset(graph_);
        if (ggml_backend_graph_compute(backend_, graph_) != GGML_STATUS_SUCCESS) throw std::runtime_error("compute failed");
        float value;
        ggml_backend_tensor_get(loss_, &value, 0, sizeof(value));
        if (!std::isfinite(value)) throw std::runtime_error("nonfinite loss");
        return value;
    }
    ggml_type type_;
    ggml_backend_t backend_ = nullptr;
    ggml_backend_buffer_t buffer_ = nullptr;
    ggml_gallocr_t allocator_ = nullptr;
    ggml_context * persistent_ = nullptr, * context_ = nullptr;
    ggml_cgraph * graph_ = nullptr;
    ggml_tensor * loss_ = nullptr;
    std::vector<Parameter> parameters_, inputs_, weights_;
    std::mt19937 rng_{831001};
};
}

int main(int argc, char ** argv) {
    try {
        const std::string device = argc>1 ? argv[1] : "cpu";
        const std::string type = argc>2 ? argv[2] : "f32";
        if (argc>3 || (type!="f32" && type!="f16" && type!="q4_k"))
            throw std::runtime_error("usage: yue2-training-probe [cpu|cuda] [f32|f16|q4_k]");
        Probe probe(device, type=="q4_k" ? GGML_TYPE_Q4_K : type=="f16" ? GGML_TYPE_F16 : GGML_TYPE_F32);
        probe.check();
        return 0;
    } catch (const std::exception & error) {
        std::cerr << "yue2-training-probe: " << error.what() << '\n';
        return 1;
    }
}
