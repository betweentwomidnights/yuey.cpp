#pragma once

#include <cstdint>
#include <vector>

namespace yue2::mert2 {

struct FrontendConfig {
    std::int32_t sample_rate = 24000;
    std::int32_t fft_size = 2048;
    std::int32_t window_size = 2048;
    std::int32_t hop_size = 240;
    std::int32_t mel_bins = 128;
    float minimum_power = 1.0e-10F;
};

struct LogMelFeatures {
    std::int64_t frames = 0;
    std::int64_t bins = 0;
    // Row-major [frames, bins], matching the Python MERT2 input to its
    // ConvNeXt subsampling module.
    std::vector<float> values;
};

// Exact CPU reference front end for the released MERT2 configuration:
// centered reflect padding, periodic Hann, power spectrum, HTK mel filters,
// 10*log10 with a 1e-10 floor, and removal of the final STFT frame.
LogMelFeatures log_mel_spectrogram(
    const std::vector<float> & waveform,
    const FrontendConfig & config = {});

} // namespace yue2::mert2

