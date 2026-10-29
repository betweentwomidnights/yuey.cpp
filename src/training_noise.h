#pragma once
#include <algorithm>
#include <cmath>
#include <random>
#include <stdexcept>
#include <string>

namespace yue2::detail {
enum class NoiseSchedule { Uniform, Sigmoid };
inline NoiseSchedule noise_schedule(const std::string &name) {
    if (name == "uniform")
        return NoiseSchedule::Uniform;
    if (name == "sigmoid")
        return NoiseSchedule::Sigmoid;
    throw std::invalid_argument("noise schedule must be uniform or sigmoid");
}
// Always consume the legacy uniform draw. A dedicated, checkpointed time RNG
// keeps the example selection and Gaussian flow noise identical between arms.
inline float training_noise_time(NoiseSchedule schedule, std::mt19937 &legacy_rng,
                                 std::mt19937 &time_rng) {
    const float uniform = std::uniform_real_distribution<float>(.05F, .95F)(legacy_rng);
    if (schedule == NoiseSchedule::Uniform)
        return uniform;
    const float z = std::normal_distribution<float>()(time_rng);
    return std::clamp(1.0F / (1.0F + std::exp(-z)), 1e-6F, 1.0F - 1e-6F);
}
} // namespace yue2::detail
