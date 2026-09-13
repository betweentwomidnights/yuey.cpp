#include "yue2/mert2_encoder.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

template <typename T>
std::vector<T> read_binary(const std::string & path) {
    std::ifstream input(path, std::ios::binary | std::ios::ate);
    if (!input) throw std::runtime_error("cannot open fixture: " + path);
    const auto bytes = input.tellg();
    if (bytes <= 0 || bytes % static_cast<std::streamoff>(sizeof(T)) != 0) {
        throw std::runtime_error("invalid fixture length: " + path);
    }
    input.seekg(0);
    std::vector<T> values(static_cast<std::size_t>(bytes / sizeof(T)));
    input.read(reinterpret_cast<char *>(values.data()), bytes);
    if (!input) throw std::runtime_error("short fixture read: " + path);
    return values;
}

} // namespace

int main(int argc, char ** argv) {
    if (argc != 5 && argc != 7) {
        std::cerr << "usage: yue2-sheetsage2-decoder-test MODEL.gguf MEMORY.f32 IDS.i32 LOGITS.f32"
                     " [GENERATED_IDS.i32 MAX_TOKENS]\n";
        return 2;
    }
    try {
        yue2::mert2::EncoderOptions options;
        options.device = std::getenv("YUE2_TEST_DEVICE") ? std::getenv("YUE2_TEST_DEVICE") : "cpu";
        yue2::mert2::Encoder model(argv[1], options);
        yue2::mert2::HiddenFeatures memory;
        memory.channels = 512;
        memory.values = read_binary<float>(argv[2]);
        if (memory.values.size() % 512 != 0) throw std::runtime_error("memory width is not 512");
        memory.frames = static_cast<std::int64_t>(memory.values.size() / 512);
        const auto ids = read_binary<std::int32_t>(argv[3]);
        const auto actual = model.decode_logits(memory, ids);
        const auto expected = read_binary<float>(argv[4]);
        if (actual.size() != expected.size()) throw std::runtime_error("logit fixture size mismatch");

        double squared_error = 0.0;
        double absolute_error = 0.0;
        double dot = 0.0, actual_norm = 0.0, expected_norm = 0.0;
        float maximum = 0.0F;
        for (std::size_t index = 0; index < actual.size(); ++index) {
            const double error = static_cast<double>(actual[index]) - expected[index];
            maximum = std::max(maximum, static_cast<float>(std::abs(error)));
            absolute_error += std::abs(error);
            squared_error += error * error;
            dot += static_cast<double>(actual[index]) * expected[index];
            actual_norm += static_cast<double>(actual[index]) * actual[index];
            expected_norm += static_cast<double>(expected[index]) * expected[index];
        }
        const double mean = absolute_error / actual.size();
        const double rms = std::sqrt(squared_error / actual.size());
        const double cosine = dot / std::sqrt(actual_norm * expected_norm);
        std::cout << "SheetSage2 decoder parity: max=" << maximum << ", mean=" << mean
                  << ", rms=" << rms << ", cosine=" << cosine << '\n';
        if (maximum > 0.08F || mean > 0.008 || rms > 0.012 || cosine < 0.99999) return 1;
        if (argc == 7) {
            const auto expected_tokens = read_binary<std::int32_t>(argv[5]);
            const auto max_tokens = static_cast<std::size_t>(std::stoull(argv[6]));
            const auto actual_tokens = model.generate_tokens(memory, ids, max_tokens);
            if (actual_tokens != expected_tokens) {
                std::cerr << "generated token mismatch\nexpected:";
                for (const auto token : expected_tokens) std::cerr << ' ' << token;
                std::cerr << "\nactual:  ";
                for (const auto token : actual_tokens) std::cerr << ' ' << token;
                std::cerr << '\n';
                return 1;
            }
            std::cout << "SheetSage2 constrained generation parity: "
                      << actual_tokens.size() << " tokens exact\n";
        }
        return 0;
    } catch (const std::exception & error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
