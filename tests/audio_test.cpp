#include "yue2/audio.h"

#include <cassert>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <iterator>
#include <vector>

namespace {

void u16(std::ofstream & stream, std::uint16_t value) {
    stream.put(static_cast<char>(value & 0xff));
    stream.put(static_cast<char>((value >> 8) & 0xff));
}

void u32(std::ofstream & stream, std::uint32_t value) {
    u16(stream, static_cast<std::uint16_t>(value & 0xffff));
    u16(stream, static_cast<std::uint16_t>(value >> 16));
}

void write_test_wav(const std::filesystem::path & path) {
    constexpr std::uint32_t rate = 48000;
    constexpr std::uint16_t channels = 2;
    constexpr std::uint32_t frames = 4800;
    constexpr std::uint32_t data_bytes = frames * channels * 2;
    std::ofstream stream(path, std::ios::binary);
    stream.write("RIFF", 4); u32(stream, 36 + data_bytes); stream.write("WAVE", 4);
    stream.write("fmt ", 4); u32(stream, 16); u16(stream, 1); u16(stream, channels);
    u32(stream, rate); u32(stream, rate * channels * 2); u16(stream, channels * 2); u16(stream, 16);
    stream.write("data", 4); u32(stream, data_bytes);
    for (std::uint32_t i = 0; i < frames; ++i) {
        const double phase = 2.0 * 3.14159265358979323846 * 440.0 * i / rate;
        const auto left = static_cast<std::int16_t>(std::sin(phase) * 12000.0);
        const auto right = static_cast<std::int16_t>(std::sin(phase) * 6000.0);
        u16(stream, static_cast<std::uint16_t>(left));
        u16(stream, static_cast<std::uint16_t>(right));
    }
}

} // namespace

int main() {
    const auto path = std::filesystem::temp_directory_path() / "yue2-audio-test.wav";
    const auto float_path = std::filesystem::temp_directory_path() / "yue2-audio-float-test.wav";
    write_test_wav(path);
    const auto original = yue2::audio::read_wav_mono(path);
    assert(original.sample_rate == 48000);
    assert(original.samples.size() == 4800);
    assert(std::abs(original.samples[1]) > 0.005F);
    std::ifstream input(path, std::ios::binary);
    const std::vector<std::uint8_t> encoded_input{
        std::istreambuf_iterator<char>(input), std::istreambuf_iterator<char>()};
    input.close();
    const auto memory_original = yue2::audio::decode_wav_mono(
        encoded_input.data(), encoded_input.size());
    assert(memory_original.sample_rate == original.sample_rate);
    assert(memory_original.samples == original.samples);

    const auto converted = yue2::audio::read_wav_mono_24k(path);
    assert(converted.sample_rate == 24000);
    assert(converted.samples.size() == 2400);
    for (float sample : converted.samples) assert(std::isfinite(sample));

    std::vector<float> stereo(2400);
    for (std::size_t frame = 0; frame < stereo.size() / 2; ++frame) {
        stereo[frame * 2] = static_cast<float>(frame) / 2400.0F;
        stereo[frame * 2 + 1] = -stereo[frame * 2];
    }
    yue2::audio::write_wav_float(float_path, stereo, 48000, 2);
    const auto float_roundtrip = yue2::audio::read_wav_mono(float_path);
    assert(float_roundtrip.sample_rate == 48000);
    assert(float_roundtrip.samples.size() == stereo.size() / 2);
    for (const auto sample : float_roundtrip.samples) assert(sample == 0.0F);
    const auto memory_float = yue2::audio::encode_wav_float(stereo, 48000, 2);
    const auto memory_roundtrip = yue2::audio::decode_wav_mono(
        memory_float.data(), memory_float.size());
    assert(memory_roundtrip.sample_rate == 48000);
    assert(memory_roundtrip.samples == float_roundtrip.samples);

    std::filesystem::remove(path);
    std::filesystem::remove(float_path);
    std::cout << "WAV decode, float encode, downmix, and 24 kHz resample: ok\n";
    return 0;
}
