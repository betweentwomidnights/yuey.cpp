#include "yue2/audio.h"
#include "yue2/mert2_encoder.h"

#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <string_view>

namespace {

std::string option(int argc, char ** argv, std::string_view name, bool required = true) {
    for (int index = 1; index < argc; ++index) {
        if (std::string_view(argv[index]) != name) continue;
        if (index + 1 >= argc) throw std::runtime_error(std::string(name) + " requires a value");
        return argv[index + 1];
    }
    if (required) throw std::runtime_error("missing " + std::string(name));
    return {};
}

bool has(int argc, char ** argv, std::string_view name) {
    for (int index = 1; index < argc; ++index) {
        if (std::string_view(argv[index]) == name) return true;
    }
    return false;
}

void usage(const char * executable) {
    std::cout
        << "Usage: " << executable
        << " --model yue2-semantic-tokenizer.gguf --audio input.wav --out tokens.i32"
           " [--device cuda|cpu] [--threads N]\n"
        << "Writes little-endian raw int32 codec IDs at 25 Hz.\n";
}

} // namespace

int main(int argc, char ** argv) {
    try {
        if (argc == 1 || has(argc, argv, "--help") || has(argc, argv, "-h")) {
            usage(argv[0]);
            return 0;
        }
        yue2::mert2::EncoderOptions options;
        options.device = option(argc, argv, "--device", false);
        const auto threads = option(argc, argv, "--threads", false);
        if (!threads.empty()) options.threads = std::stoi(threads);
        auto audio = yue2::audio::read_wav_mono_24k(option(argc, argv, "--audio"));
        yue2::mert2::Encoder encoder(option(argc, argv, "--model"), options);
        const auto tokens = encoder.semantic_tokens_24k(audio.samples);

        const std::filesystem::path output = option(argc, argv, "--out");
        if (!output.parent_path().empty()) {
            std::filesystem::create_directories(output.parent_path());
        }
        std::ofstream stream(output, std::ios::binary);
        if (!stream || !stream.write(
                reinterpret_cast<const char *>(tokens.data()),
                static_cast<std::streamsize>(tokens.size() * sizeof(tokens.front())))) {
            throw std::runtime_error("could not write " + output.string());
        }
        std::cout << "wrote " << output << " (" << tokens.size()
                  << " semantic frames, " << tokens.size() / 25.0 << " seconds)\n";
        return 0;
    } catch (const std::exception & error) {
        std::cerr << "yue2-semantic-tokenize: " << error.what() << '\n';
        return 2;
    }
}
