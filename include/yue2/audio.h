#pragma once

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <vector>

namespace yue2::audio {

constexpr std::int32_t transcription_sample_rate = 24000;

struct MonoAudio {
    std::vector<float> samples;
    std::int32_t sample_rate = 0;
};

// Reads PCM16/24/32 or IEEE-float32/64 RIFF/WAVE, averages every channel, and
// preserves sample amplitude. It intentionally performs no peak normalization.
MonoAudio read_wav_mono(const std::filesystem::path & path);

// Decode a complete RIFF/WAVE payload already owned by a host or HTTP upload.
MonoAudio decode_wav_mono(const std::uint8_t * bytes, std::size_t byte_count);

std::vector<float> resample_sinc(
    const std::vector<float> & input,
    std::int32_t input_rate,
    std::int32_t output_rate);

MonoAudio read_wav_mono_24k(const std::filesystem::path & path);

// Writes interleaved IEEE-float32 PCM as a standard little-endian RIFF/WAVE.
// The function preserves model output amplitude and rejects non-finite samples.
void write_wav_float(
    const std::filesystem::path & path,
    const std::vector<float> & interleaved_samples,
    std::int32_t sample_rate,
    std::int32_t channels);

// Encode the same IEEE-float32 RIFF/WAVE representation entirely in memory.
std::vector<std::uint8_t> encode_wav_float(
    const std::vector<float> & interleaved_samples,
    std::int32_t sample_rate,
    std::int32_t channels);

} // namespace yue2::audio
