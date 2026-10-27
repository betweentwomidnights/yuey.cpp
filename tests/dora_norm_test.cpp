// Feasibility oracle for YuE2 DoRA primitives. Not a trained adapter or loader.
// Compare the low-rank norm identity against an independent dense computation
// and numerical gradients, including Toolkit's detached-denominator convention.
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml.h"
#include "dora.h"
#include <algorithm>
#include <array>
#include <cassert>
#include <cmath>
#include <iostream>
#include <random>
#include <vector>

namespace {
constexpr int I = 9, O = 7, R = 3, T = 5;
constexpr float S = .7F;
using V = std::vector<float>;
struct Data {
    V w, a, b, m, x, target;
};
V norms(const Data &d) {
    V result(O);
    for (int o = 0; o < O; ++o) {
        double sum = 0;
        for (int i = 0; i < I; ++i) {
            double weight = d.w[o * I + i];
            for (int r = 0; r < R; ++r)
                weight += S * double(d.b[o * R + r]) * d.a[r * I + i];
            sum += weight * weight;
        }
        result[o] = float(std::sqrt(sum + 1e-12));
    }
    return result;
}
double dense_loss(const Data &d, const V *detached = nullptr, V *output = nullptr) {
    auto norm = detached ? *detached : norms(d);
    double loss = 0;
    if (output)
        output->resize(O * T);
    for (int t = 0; t < T; ++t)
        for (int o = 0; o < O; ++o) {
            double y = 0;
            for (int i = 0; i < I; ++i) {
                double weight = d.w[o * I + i];
                for (int r = 0; r < R; ++r)
                    weight += S * double(d.b[o * R + r]) * d.a[r * I + i];
                y += weight * d.x[t * I + i];
            }
            y *= d.m[o] / norm[o];
            if (output)
                (*output)[t * O + o] = float(y);
            const double delta = y - d.target[t * O + o];
            loss += delta * delta / (O * T);
        }
    return loss;
}
double check(ggml_backend_t backend, Data d, bool detach) {
    auto *ctx = ggml_init({16 * 1024 * 1024, nullptr, true});
    assert(ctx);
    auto tensor = [&](const V &v, int n, int m = 1, bool parameter = false) {
        auto *p = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, n, m);
        ggml_set_input(p);
        if (parameter)
            ggml_set_param(p);
        return p;
    };
    auto *w = tensor(d.w, I, O), *a = tensor(d.a, I, R, true), *b = tensor(d.b, R, O, true);
    auto *m = tensor(d.m, O, 1, true), *x = tensor(d.x, I, T, true),
         *target = tensor(d.target, O, T);
    V base(O);
    for (int o = 0; o < O; ++o)
        for (int i = 0; i < I; ++i)
            base[o] += d.w[o * I + i] * d.w[o * I + i];
    auto *bn = tensor(base, O);
    auto *y = ggml_add(ctx, ggml_mul_mat(ctx, w, x),
                       ggml_scale(ctx, ggml_mul_mat(ctx, b, ggml_mul_mat(ctx, a, x)), S));
    auto *denom = yue2::detail::dora_row_norm(ctx,w,a,b,ggml_reshape_1d(ctx,bn,O),S);
    const auto fixed_norm = norms(d);
    ggml_tensor *dn = nullptr;
    if (detach) {
        dn = tensor(fixed_norm, O);
        denom = ggml_reshape_1d(ctx, dn, O);
    }
    y = ggml_mul(ctx, y, ggml_div(ctx, ggml_reshape_1d(ctx, m, O), denom));
    ggml_set_output(y);
    auto *loss =
        ggml_scale(ctx, ggml_sum(ctx, ggml_sqr(ctx, ggml_sub(ctx, y, target))), 1.F / (O * T));
    ggml_set_loss(loss);
    ggml_set_output(loss);
    auto *graph = ggml_new_graph_custom(ctx, 2048, true);
    ggml_build_forward_expand(graph, loss);
    ggml_build_backward_expand(ctx, graph, nullptr);
    const std::array<ggml_tensor *, 4> params{a, b, m, x};
    for (auto *p : params) {
        assert(ggml_graph_get_grad(graph, p));
        ggml_set_output(ggml_graph_get_grad(graph, p));
    }
    auto *alloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
    assert(ggml_gallocr_alloc_graph(alloc, graph));
    auto upload = [](ggml_tensor *p, const V &v) {
        ggml_backend_tensor_set(p, v.data(), 0, v.size() * sizeof(float));
    };
    upload(w, d.w);
    upload(a, d.a);
    upload(b, d.b);
    upload(m, d.m);
    upload(x, d.x);
    upload(target, d.target);
    // The detached branch does not reference the low-rank norm subgraph.
    if (!detach)
        upload(bn, base);
    else
        upload(dn, fixed_norm);
    ggml_graph_reset(graph);
    assert(ggml_backend_graph_compute(backend, graph) == GGML_STATUS_SUCCESS);
    V expected;
    const auto expected_loss = dense_loss(d, detach ? &fixed_norm : nullptr, &expected);
    float actual_loss;
    ggml_backend_tensor_get(loss, &actual_loss, 0, sizeof(float));
    assert(std::abs(actual_loss - expected_loss) < 2e-5);
    V actual(expected.size());
    ggml_backend_tensor_get(y, actual.data(), 0, actual.size() * sizeof(float));
    for (size_t i = 0; i < actual.size(); ++i)
        assert(std::abs(actual[i] - expected[i]) < 2e-5);
    std::array<V *, 4> values{&d.a, &d.b, &d.m, &d.x};
    double worst = 0;
    for (size_t p = 0; p < params.size(); ++p) {
        V grad(values[p]->size());
        ggml_backend_tensor_get(ggml_graph_get_grad(graph, params[p]), grad.data(), 0,
                                grad.size() * sizeof(float));
        for (size_t i = 0; i < grad.size(); ++i) {
            auto &v = (*values[p])[i];
            const float old = v;
            v = old + .002F;
            const auto plus = dense_loss(d, detach ? &fixed_norm : nullptr);
            const float hi = v;
            v = old - .002F;
            const auto minus = dense_loss(d, detach ? &fixed_norm : nullptr);
            const float lo = v;
            v = old;
            const auto expected_grad = (plus - minus) / (hi - lo);
            const auto error = std::abs(grad[i] - expected_grad);
            worst = std::max(worst, error);
            assert(error < 3e-4 + 2e-3 * std::abs(expected_grad));
        }
    }
    ggml_gallocr_free(alloc);
    ggml_free(ctx);
    return worst;
}
} // namespace
int main(int argc, char **argv) {
    ggml_backend_load_all();
    auto *backend = ggml_backend_init_by_name(argc > 1 ? argv[1] : "CPU", nullptr);
    assert(backend);
    std::mt19937 rng(712);
    std::uniform_real_distribution<float> random(-.6F, .6F);
    auto make = [&](size_t n) {
        V v(n);
        for (auto &x : v)
            x = random(rng);
        return v;
    };
    Data d{make(I * O), make(I * R), make(R * O), V(O, 1), make(I * T), make(O * T)};
    const auto full = check(backend, d, false), detached = check(backend, d, true);
    std::fill(d.b.begin(), d.b.end(), 0.0F);
    d.m = norms(d);
    check(backend, d, false);
    V identity;
    dense_loss(d, nullptr, &identity);
    for (int t = 0; t < T; ++t)
        for (int o = 0; o < O; ++o) {
            double y = 0;
            for (int i = 0; i < I; ++i)
                y += d.w[o * I + i] * d.x[t * I + i];
            assert(std::abs(y - identity[t * O + o]) < 1e-6);
        }
    std::cout << "{\"passed\":true,\"full_norm_max_gradient_error\":" << full
              << ",\"detached_norm_max_gradient_error\":" << detached
              << ",\"zero_delta_identity\":true}\n";
    ggml_backend_free(backend);
}
