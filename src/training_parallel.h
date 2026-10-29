#pragma once

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <mutex>
#include <thread>
#include <vector>

namespace yue2::detail {
// Run work(task) for every task in [0,count) on up to `threads` threads.
// Tasks must touch disjoint data: elementwise results are then bit-identical
// to a serial loop. Reductions must not be split across tasks. The first
// exception is rethrown after every worker has stopped.
template <class Work> void parallel_tasks(size_t count, int threads, Work &&work) {
    const auto workers = std::min<size_t>(static_cast<size_t>(std::max(threads, 1)), count);
    if (workers <= 1) {
        for (size_t task = 0; task < count; ++task)
            work(task);
        return;
    }
    std::atomic<size_t> next{0};
    std::exception_ptr error;
    std::mutex lock;
    auto run = [&] {
        try {
            for (size_t task; (task = next++) < count;)
                work(task);
        } catch (...) {
            std::lock_guard<std::mutex> guard(lock);
            if (!error)
                error = std::current_exception();
            next = count;
        }
    };
    std::vector<std::thread> pool;
    pool.reserve(workers - 1);
    for (size_t i = 1; i < workers; ++i)
        pool.emplace_back(run);
    run();
    for (auto &thread : pool)
        thread.join();
    if (error)
        std::rethrow_exception(error);
}

// Elementwise span of one optimizer tensor.
struct TensorSpan {
    size_t tensor, begin, end;
};
// Fixed-size spans over every tensor, in tensor order, for parallel_tasks.
template <class Tensors> std::vector<TensorSpan> tensor_spans(const Tensors &tensors, size_t span) {
    std::vector<TensorSpan> result;
    for (size_t t = 0; t < tensors.size(); ++t)
        for (size_t begin = 0; begin < tensors[t].values.size(); begin += span)
            result.push_back({t, begin, std::min(tensors[t].values.size(), begin + span)});
    return result;
}

inline bool all_finite(const float *values, size_t begin, size_t end) {
    bool ok = true;
    for (size_t i = begin; i < end; ++i)
        ok &= std::isfinite(values[i]);
    return ok;
}

// The trainers' AdamW update for elements [begin,end). Expression and operand
// types are exactly the former serial loop's; each element is independent.
// Returns false when an updated value is nonfinite.
inline bool adamw_span(float *values, const float *gradient, float *m, float *v, size_t begin,
                       size_t end, double scale, double correction1, double correction2,
                       float learning_rate, float weight_decay) {
    for (size_t i = begin; i < end; ++i) {
        const float g = static_cast<float>(gradient[i] * scale);
        m[i] = 0.9F * m[i] + 0.1F * g;
        v[i] = 0.999F * v[i] + 0.001F * g * g;
        values[i] -= learning_rate *
                     static_cast<float>((m[i] / correction1) / (std::sqrt(v[i] / correction2) + 1e-8) +
                                        weight_decay * values[i]);
    }
    return all_finite(values, begin, end);
}
// IEEE half: exponent all ones is infinity or NaN.
inline bool all_finite_half(const uint16_t *values, size_t begin, size_t end) {
    uint16_t any = 0;
    for (size_t i = begin; i < end; ++i)
        any |= static_cast<uint16_t>((values[i] & 0x7c00) == 0x7c00);
    return !any;
}
} // namespace yue2::detail
