#include "yue2/mert2_frontend.h"

#include <array>
#include <cassert>
#include <cmath>
#include <cstddef>
#include <iostream>
#include <vector>

int main() {
    constexpr double pi = 3.14159265358979323846;
    std::vector<float> waveform(2400);
    for (std::size_t i = 0; i < waveform.size(); ++i) {
        waveform[i] = static_cast<float>(
            0.2 * std::sin(2.0 * pi * 440.0 * i / 24000.0) +
            0.05 * std::cos(2.0 * pi * 997.0 * i / 24000.0));
    }
    const auto features = yue2::mert2::log_mel_spectrogram(waveform);
    assert(features.frames == 10);
    assert(features.bins == 128);

    struct Probe { std::size_t frame; std::size_t bin; float expected; };
    constexpr std::array<Probe, 8> probes{{
        {0, 0, 11.8709698F}, {0, 1, 12.3160934F}, {0, 64, -9.8421955F},
        {0, 127, -24.9804916F}, {1, 0, 10.6658840F}, {1, 64, -11.0470428F},
        {5, 10, -46.8706169F}, {9, 127, -22.8231812F},
    }};
    for (const auto & probe : probes) {
        const float actual = features.values[probe.frame * 128 + probe.bin];
        if (std::abs(actual - probe.expected) > 2.0e-3F) {
            std::cerr << "log-mel mismatch at [" << probe.frame << ',' << probe.bin
                      << "]: expected " << probe.expected << ", got " << actual << '\n';
            return 1;
        }
    }
    std::cout << "MERT2 log-mel frontend parity: ok\n";
    return 0;
}

