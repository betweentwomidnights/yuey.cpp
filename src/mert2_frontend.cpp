#include "yue2/mert2_frontend.h"

#include <algorithm>
#include <atomic>
#include <cmath>
#include <complex>
#include <cstddef>
#include <stdexcept>
#include <thread>

namespace yue2::mert2 {
namespace {

constexpr double pi = 3.1415926535897932384626433832795;

double hz_to_mel(double frequency) {
    return 2595.0 * std::log10(1.0 + frequency / 700.0);
}

double mel_to_hz(double mel) {
    return 700.0 * (std::pow(10.0, mel / 2595.0) - 1.0);
}

std::size_t reflected_index(std::int64_t position, std::size_t length) {
    const auto end = static_cast<std::int64_t>(length) - 1;
    while (position < 0 || position > end) {
        if (position < 0) position = -position;
        if (position > end) position = 2 * end - position;
    }
    return static_cast<std::size_t>(position);
}

void fft(std::vector<std::complex<double>> & values) {
    const std::size_t n = values.size();
    for (std::size_t i = 1, j = 0; i < n; ++i) {
        std::size_t bit = n >> 1;
        for (; (j & bit) != 0; bit >>= 1) j ^= bit;
        j ^= bit;
        if (i < j) std::swap(values[i], values[j]);
    }
    for (std::size_t length = 2; length <= n; length <<= 1) {
        const double angle = -2.0 * pi / length;
        const std::complex<double> root(std::cos(angle), std::sin(angle));
        for (std::size_t base = 0; base < n; base += length) {
            std::complex<double> factor(1.0, 0.0);
            for (std::size_t i = 0; i < length / 2; ++i) {
                const auto even = values[base + i];
                const auto odd = values[base + i + length / 2] * factor;
                values[base + i] = even + odd;
                values[base + i + length / 2] = even - odd;
                factor *= root;
            }
        }
    }
}

struct FilterEntry {
    std::int32_t frequency_bin;
    float weight;
};

std::vector<std::vector<FilterEntry>> make_mel_filters(const FrontendConfig & config) {
    const std::int32_t frequencies = config.fft_size / 2 + 1;
    std::vector<double> points(static_cast<std::size_t>(config.mel_bins + 2));
    const double mel_min = hz_to_mel(0.0);
    const double mel_max = hz_to_mel(config.sample_rate / 2.0);
    for (std::int32_t i = 0; i < config.mel_bins + 2; ++i) {
        const double mel = mel_min + (mel_max - mel_min) * i / (config.mel_bins + 1);
        points[static_cast<std::size_t>(i)] = mel_to_hz(mel);
    }

    std::vector<std::vector<FilterEntry>> filters(static_cast<std::size_t>(config.mel_bins));
    for (std::int32_t frequency_bin = 0; frequency_bin < frequencies; ++frequency_bin) {
        const double frequency = (config.sample_rate / 2.0) * frequency_bin / (frequencies - 1);
        for (std::int32_t mel = 0; mel < config.mel_bins; ++mel) {
            const double left = points[static_cast<std::size_t>(mel)];
            const double center = points[static_cast<std::size_t>(mel + 1)];
            const double right = points[static_cast<std::size_t>(mel + 2)];
            const double down = (frequency - left) / (center - left);
            const double up = (right - frequency) / (right - center);
            const float weight = static_cast<float>(std::max(0.0, std::min(down, up)));
            if (weight > 0.0F) filters[static_cast<std::size_t>(mel)].push_back({frequency_bin, weight});
        }
    }
    return filters;
}

} // namespace

LogMelFeatures log_mel_spectrogram(
    const std::vector<float> & waveform,
    const FrontendConfig & config) {
    if (config.sample_rate <= 0 || config.fft_size <= 0 || config.window_size <= 0 ||
        config.window_size > config.fft_size || config.hop_size <= 0 || config.mel_bins <= 0 ||
        config.minimum_power <= 0.0F || (config.fft_size & (config.fft_size - 1)) != 0) {
        throw std::invalid_argument("invalid MERT2 frontend configuration");
    }
    if (waveform.size() <= static_cast<std::size_t>(config.fft_size / 2)) {
        throw std::invalid_argument("MERT2 waveform is too short for reflect padding");
    }
    for (float sample : waveform) {
        if (!std::isfinite(sample)) throw std::invalid_argument("MERT2 waveform contains non-finite samples");
    }

    // torch.stft(center=True) has one frame per hop plus the initial frame. The
    // released MERT2 front end then removes the final frame.
    const std::int64_t stft_frames = static_cast<std::int64_t>(waveform.size() / config.hop_size) + 1;
    const std::int64_t output_frames = stft_frames - 1;
    LogMelFeatures result;
    result.frames = output_frames;
    result.bins = config.mel_bins;
    result.values.resize(static_cast<std::size_t>(output_frames * config.mel_bins));

    std::vector<float> window(static_cast<std::size_t>(config.window_size));
    for (std::int32_t i = 0; i < config.window_size; ++i) {
        window[static_cast<std::size_t>(i)] = static_cast<float>(0.5 * (1.0 - std::cos(2.0 * pi * i / config.window_size)));
    }
    const auto filters = make_mel_filters(config);
    const std::int64_t padding = config.fft_size / 2;
    const std::int64_t window_offset = (config.fft_size - config.window_size) / 2;

    std::atomic<std::int64_t> next_frame{0};
    const auto hardware = std::max(1U, std::thread::hardware_concurrency());
    const auto workers = static_cast<unsigned>(std::min<std::int64_t>(hardware, output_frames));
    std::vector<std::thread> threads;
    threads.reserve(workers);
    for (unsigned worker = 0; worker < workers; ++worker) {
        threads.emplace_back([&]() {
            std::vector<std::complex<double>> spectrum(static_cast<std::size_t>(config.fft_size));
            std::vector<double> power(static_cast<std::size_t>(config.fft_size / 2 + 1));
            for (;;) {
                const auto frame = next_frame.fetch_add(1);
                if (frame >= output_frames) break;
                std::fill(spectrum.begin(), spectrum.end(), std::complex<double>{});
                const std::int64_t start = frame * config.hop_size - padding + window_offset;
                for (std::int32_t i = 0; i < config.window_size; ++i) {
                    const auto source = reflected_index(start + i, waveform.size());
                    spectrum[static_cast<std::size_t>(window_offset + i)] = waveform[source] * window[static_cast<std::size_t>(i)];
                }
                fft(spectrum);
                for (std::size_t i = 0; i < power.size(); ++i) power[i] = std::norm(spectrum[i]);
                for (std::int32_t mel = 0; mel < config.mel_bins; ++mel) {
                    double value = 0.0;
                    for (const auto & entry : filters[static_cast<std::size_t>(mel)]) {
                        value += power[static_cast<std::size_t>(entry.frequency_bin)] * entry.weight;
                    }
                    value = std::max<double>(value, config.minimum_power);
                    result.values[static_cast<std::size_t>(frame * config.mel_bins + mel)] =
                        static_cast<float>(10.0 * std::log10(value));
                }
            }
        });
    }
    for (auto & thread : threads) thread.join();
    return result;
}

} // namespace yue2::mert2
