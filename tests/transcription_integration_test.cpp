#include "yue2/audio.h"
#include "yue2/transcription.h"

#include <cassert>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <future>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

void u16(std::ofstream & stream, std::uint16_t value) {
    stream.put(static_cast<char>(value & 0xffU));
    stream.put(static_cast<char>((value >> 8U) & 0xffU));
}

void u32(std::ofstream & stream, std::uint32_t value) {
    u16(stream, static_cast<std::uint16_t>(value & 0xffffU));
    u16(stream, static_cast<std::uint16_t>(value >> 16U));
}

void write_test_wav(const std::filesystem::path & path) {
    constexpr std::uint32_t rate = 24000;
    constexpr std::uint32_t frames = 4800;
    constexpr std::uint32_t data_bytes = frames * 2;
    std::ofstream stream(path, std::ios::binary);
    if (!stream) throw std::runtime_error("cannot create integration WAV");
    stream.write("RIFF", 4); u32(stream, 36 + data_bytes); stream.write("WAVE", 4);
    stream.write("fmt ", 4); u32(stream, 16); u16(stream, 1); u16(stream, 1);
    u32(stream, rate); u32(stream, rate * 2); u16(stream, 2); u16(stream, 16);
    stream.write("data", 4); u32(stream, data_bytes);
    for (std::uint32_t index = 0; index < frames; ++index) {
        const double phase = 2.0 * 3.14159265358979323846 * 220.0 * index / rate;
        const auto sample = static_cast<std::int16_t>(std::sin(phase) * 8000.0);
        u16(stream, static_cast<std::uint16_t>(sample));
    }
}

struct RemoveFile {
    std::filesystem::path path;
    bool active = true;
    ~RemoveFile() {
        if (!active) return;
        std::error_code error;
        std::filesystem::remove(path, error);
    }
};

} // namespace

int main(int argc, char ** argv) {
    if (argc < 2 || argc > 3) {
        std::cerr << "usage: yue2-transcription-integration-test MODEL.gguf [KEEP-WAV]\n";
        return 2;
    }
    const auto wav_path = argc == 3
        ? std::filesystem::path(argv[2])
        : std::filesystem::temp_directory_path() / "yue2-transcription-integration.wav";
    const RemoveFile cleanup{wav_path, argc == 2};
    write_test_wav(wav_path);

    yue2::TranscriptionOptions options;
    options.window_seconds = 0.2F;
    options.max_tokens = 6;
    yue2::Transcriber transcriber(argv[1]);
    const auto result = transcriber.transcribe(wav_path, options);
    const auto pcm = yue2::audio::read_wav_mono(wav_path);
    const auto pcm_result = transcriber.transcribe_mono(
        pcm.samples.data(), pcm.samples.size(), pcm.sample_rate, options);

    assert(yue2::transcription_runtime_available());
    assert(std::string(yue2::version()).find("dev") != std::string::npos);
    assert(std::abs(result.duration_seconds - 0.2) < 1e-6);
    assert(result.abc.rfind("X:1\n", 0) == 0);
    assert(result.abc.find("V: Vocal\n") != std::string::npos);
    assert(result.abc.find("V: Ins\n") != std::string::npos);
    assert(result.midi.size() >= 22);
    assert(std::string(result.midi.begin(), result.midi.begin() + 4) == "MThd");
    assert(result.events.size() == 1);
    assert(result.tokens.size() == 7 && result.tokens.back() == 2);
    assert(pcm_result.tokens == result.tokens);
    assert(pcm_result.abc == result.abc);
    assert(pcm_result.midi == result.midi);
    assert(result.events.front().has_timestamp);
    assert(result.events.front().time_seconds == 0.0);
    assert(!result.warnings.empty());

    auto first_call = std::async(std::launch::async, [&]() {
        return transcriber.transcribe_mono(
            pcm.samples.data(), pcm.samples.size(), pcm.sample_rate, options);
    });
    auto second_call = std::async(std::launch::async, [&]() {
        return transcriber.transcribe_mono(
            pcm.samples.data(), pcm.samples.size(), pcm.sample_rate, options);
    });
    assert(first_call.get().tokens == result.tokens);
    assert(second_call.get().tokens == result.tokens);

    std::vector<float> long_pcm(48000);
    for (std::size_t index = 0; index < long_pcm.size(); ++index) {
        long_pcm[index] = static_cast<float>(
            std::sin(2.0 * 3.14159265358979323846 * 220.0 * index / 24000.0) * 0.25);
    }
    yue2::TranscriptionOptions long_options;
    long_options.window_seconds = 1.0F;
    long_options.overlap_seconds = 0.6F;
    long_options.lookahead_seconds = 0.3F;
    long_options.max_tokens = 32;
    yue2::TranscriptionResult long_result;
    try {
        long_result = transcriber.transcribe_mono(
            long_pcm.data(), long_pcm.size(), 24000, long_options);
    } catch (const std::exception & error) {
        std::cerr << "whole-song transcription failed: " << error.what() << '\n';
        return 1;
    }
    assert(long_result.windows.size() == 4);
    assert(long_result.tokens.empty());
    assert(long_result.windows[1].prefix_tokens > 4);
    for (std::size_t index = 1; index < long_result.events.size(); ++index) {
        assert(long_result.events[index - 1].time_seconds <= long_result.events[index].time_seconds);
    }
    std::cout << "single-window transcription API: ok ("
              << result.events.size() << " decoded events); whole-song stitching: ok ("
              << long_result.windows.size() << " windows)\n";
    return 0;
}
