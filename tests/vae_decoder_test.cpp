#include "yue2/vae.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <future>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

std::vector<float> read_f32(const std::string & path) {
    std::ifstream stream(path, std::ios::binary | std::ios::ate);
    if (!stream) throw std::runtime_error("cannot open fixture: " + path);
    const auto bytes = stream.tellg();
    if (bytes < 0 || bytes % static_cast<std::streamoff>(sizeof(float)) != 0) {
        throw std::runtime_error("invalid f32 fixture size: " + path);
    }
    std::vector<float> values(static_cast<std::size_t>(bytes) / sizeof(float));
    stream.seekg(0);
    stream.read(reinterpret_cast<char *>(values.data()), bytes);
    if (!stream) throw std::runtime_error("short fixture read: " + path);
    return values;
}

} // namespace

int main(int argc, char ** argv) {
    if (argc < 5 || argc > 7) {
        std::cerr << "usage: yue2-vae-decoder-test VAE.gguf LATENTS.f32 EXPECTED.f32 FRAMES [DEVICE] [full]\n";
        return 2;
    }
    try {
        const auto frames = std::stoll(argv[4]);
        const auto latents = read_f32(argv[2]);
        const auto expected = read_f32(argv[3]);
        if (frames <= 0 || latents.size() != static_cast<std::size_t>(frames * 64)) {
            throw std::runtime_error("latent fixture shape mismatch");
        }
        yue2::VaeRuntimeOptions options;
        if (argc >= 6) options.device = argv[5];
        if (argc == 7) {
            if (std::string(argv[6]) != "full") throw std::runtime_error("expected final argument 'full'");
            options.tiled = false;
        }
        yue2::VaeDecoder decoder(argv[1], options);
        const auto actual = decoder.decode(latents);
        if (actual.sample_rate != 48000 || actual.channels != 2 ||
            actual.interleaved_samples.size() != expected.size()) {
            throw std::runtime_error("decoded audio shape mismatch");
        }

        double absolute_sum = 0.0;
        double squared_sum = 0.0;
        double maximum = 0.0;
        std::size_t maximum_index = 0;
        for (std::size_t index = 0; index < expected.size(); ++index) {
            const double error = std::abs(
                static_cast<double>(actual.interleaved_samples[index]) - expected[index]);
            absolute_sum += error;
            squared_sum += error * error;
            if (error > maximum) {
                maximum = error;
                maximum_index = index;
            }
        }
        const double mean = absolute_sum / expected.size();
        const double rms = std::sqrt(squared_sum / expected.size());
        std::cout << "VAE samples=" << expected.size() / 2
                  << " max=" << maximum << "@sample=" << maximum_index / 2
                  << ",channel=" << maximum_index % 2
                  << " mean=" << mean << " rms=" << rms << '\n';
        // The SnakeBeta stack amplifies sparse backend rounding differences on
        // long signals. Keep the aggregate bound tight while allowing those
        // measured CUDA/CPU peaks; short fixtures retain a strict peak bound.
        const double maximum_tolerance = frames <= 64 ? 0.003 : 0.1;
        const double rms_tolerance = frames <= 64 ? 0.001 : 0.0012;
        if (maximum > maximum_tolerance || rms > rms_tolerance) {
            throw std::runtime_error("VAE parity tolerance exceeded");
        }
        if (frames <= 2) {
            auto first = std::async(std::launch::async, [&]() { return decoder.decode(latents); });
            auto second = std::async(std::launch::async, [&]() { return decoder.decode(latents); });
            if (first.get().interleaved_samples != actual.interleaved_samples ||
                second.get().interleaved_samples != actual.interleaved_samples) {
                throw std::runtime_error("concurrent VAE decode was not deterministic");
            }
            std::cout << "VAE same-instance concurrency: ok\n";
        }
        return 0;
    } catch (const std::exception & error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
