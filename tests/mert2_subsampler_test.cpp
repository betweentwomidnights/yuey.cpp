#include "yue2/mert2_encoder.h"
#include "yue2/mert2_frontend.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

std::vector<float> make_waveform(std::size_t sample_count) {
    constexpr double pi = 3.1415926535897932384626433832795;
    std::vector<float> values(sample_count);
    for (std::size_t index = 0; index < sample_count; ++index) {
        const double signal =
            0.10 * std::sin(2.0 * pi * 311.0 * index / 24000.0) +
            0.05 * std::cos(2.0 * pi * 997.0 * index / 24000.0) +
            0.002 * ((static_cast<double>(index % 97) - 48.0) / 48.0);
        values[index] = static_cast<float>(signal);
    }
    return values;
}

std::vector<float> read_f32(const std::string & path) {
    std::ifstream input(path, std::ios::binary | std::ios::ate);
    if (!input) throw std::runtime_error("cannot open reference: " + path);
    const auto bytes = input.tellg();
    if (bytes <= 0 || bytes % static_cast<std::streamoff>(sizeof(float)) != 0) {
        throw std::runtime_error("invalid reference length");
    }
    input.seekg(0);
    std::vector<float> result(static_cast<std::size_t>(bytes / sizeof(float)));
    input.read(reinterpret_cast<char *>(result.data()), bytes);
    if (!input) throw std::runtime_error("short read from reference");
    return result;
}

} // namespace

bool compare(
    const char * label,
    const yue2::mert2::HiddenFeatures & actual,
    const std::vector<float> & expected,
    float maximum_tolerance,
    double mean_tolerance,
    double rms_tolerance) {
    if (actual.channels != 1024 || actual.frames != 5 || actual.values.size() != expected.size()) {
        std::cerr << label << " shape mismatch: native=[" << actual.frames << ',' << actual.channels
                  << "] reference-values=" << expected.size() << '\n';
        return false;
    }
    double absolute_sum = 0.0;
    double squared_sum = 0.0;
    float maximum = 0.0F;
    std::size_t maximum_index = 0;
    for (std::size_t index = 0; index < expected.size(); ++index) {
        if (!std::isfinite(actual.values[index])) {
            std::cerr << "non-finite " << label << " value at " << index << '\n';
            return false;
        }
        const float error = std::abs(actual.values[index] - expected[index]);
        absolute_sum += error;
        squared_sum += static_cast<double>(error) * error;
        if (error > maximum) {
            maximum = error;
            maximum_index = index;
        }
    }
    const double mean = absolute_sum / expected.size();
    const double rms = std::sqrt(squared_sum / expected.size());
    std::cout << label << " parity: max=" << maximum << " at " << maximum_index
              << ", mean=" << mean << ", rms=" << rms << '\n';
    if (maximum > maximum_tolerance || mean > mean_tolerance || rms > rms_tolerance) {
        std::cerr << label << " exceeds parity tolerance\n";
        return false;
    }
    return true;
}

int main(int argc, char ** argv) {
    if (argc < 3 || argc > 6) {
        std::cerr << "usage: yue2-mert2-subsampler-test MODEL.gguf SUBSAMPLED.f32 "
                     "[ENCODED.f32 [MEMORY.f32] | ENCODED.f32 LAYER_COUNT LAYER.f32]\n";
        return 2;
    }
    try {
        const char * test_device = std::getenv("YUE2_TEST_DEVICE");
        const bool accelerator_test = test_device && std::string(test_device) != "cpu";
        const auto features = yue2::mert2::log_mel_spectrogram(make_waveform(4800));
        yue2::mert2::EncoderOptions options;
        options.device = test_device ? test_device : "cpu";
        yue2::mert2::Encoder encoder(argv[1], options);
        const auto actual = encoder.subsample(features);
        if (const char * dump_path = std::getenv("YUE2_DUMP_SUBSAMPLED")) {
            std::ofstream dump(dump_path, std::ios::binary);
            dump.write(reinterpret_cast<const char *>(actual.values.data()),
                static_cast<std::streamsize>(actual.values.size() * sizeof(float)));
            if (!dump) throw std::runtime_error("could not dump native subsampler output");
        }
        const auto expected = read_f32(argv[2]);
        // F16 matrix storage compounds over twelve ConvNeXt layers. These
        // bounds are intentionally strict enough to catch layout/operator
        // mistakes while accepting expected conversion error.
        if (!compare("MERT2 subsampler", actual, expected,
                accelerator_test ? 0.1F : 0.03F,
                accelerator_test ? 0.01 : 0.004,
                accelerator_test ? 0.013 : 0.006)) return 1;
        if (argc >= 4) {
            const auto encoded = encoder.encode(features);
            const auto encoded_expected = read_f32(argv[3]);
            if (!compare("MERT2 encoder", encoded, encoded_expected, 0.09F, 0.005, 0.007)) return 1;
        }
        if (argc == 5) {
            const auto memory = encoder.sheetsage_memory(features);
            const auto expected_memory = read_f32(argv[4]);
            if (memory.channels != 512) {
                std::cerr << "SheetSage2 memory channel mismatch\n";
                return 1;
            }
            // The common comparison expects 1024 channels, so compare the
            // projected memory inline with its actual [5,512] shape.
            yue2::mert2::HiddenFeatures padded = memory;
            padded.channels = 1024;
            padded.frames = expected_memory.size() / 1024;
            if (memory.values.size() != expected_memory.size()) {
                std::cerr << "SheetSage2 memory size mismatch\n";
                return 1;
            }
            double sum = 0.0, sum2 = 0.0;
            float maximum = 0.0F;
            float native_minimum = std::numeric_limits<float>::infinity();
            float native_maximum = -std::numeric_limits<float>::infinity();
            double native_sum = 0.0;
            for (std::size_t index = 0; index < memory.values.size(); ++index) {
                const float error = std::abs(memory.values[index] - expected_memory[index]);
                maximum = std::max(maximum, error);
                sum += error;
                sum2 += static_cast<double>(error) * error;
                native_minimum = std::min(native_minimum, memory.values[index]);
                native_maximum = std::max(native_maximum, memory.values[index]);
                native_sum += memory.values[index];
            }
            const double mean = sum / memory.values.size();
            const double rms = std::sqrt(sum2 / memory.values.size());
            std::cout << "SheetSage2 memory parity: max=" << maximum << ", mean=" << mean << ", rms=" << rms
                      << ", native-range=[" << native_minimum << ',' << native_maximum << "], native-mean="
                      << native_sum / memory.values.size() << '\n';
            if (maximum > 0.05F || mean > 0.004 || rms > 0.006) return 1;
        }
        if (argc == 6) {
            const int layer_count = std::stoi(argv[4]);
            const auto layer = layer_count < 0
                ? encoder.layer0_stage(features, -layer_count)
                : encoder.encode(features, layer_count);
            if (const char * dump_path = std::getenv("YUE2_DUMP_RESULT")) {
                std::ofstream dump(dump_path, std::ios::binary);
                dump.write(reinterpret_cast<const char *>(layer.values.data()),
                    static_cast<std::streamsize>(layer.values.size() * sizeof(float)));
                if (!dump) throw std::runtime_error("could not dump diagnostic result");
            }
            if (!compare("MERT2 diagnostic layer", layer, read_f32(argv[5]), 0.2F, 0.03, 0.04)) return 1;
        }
        return 0;
    } catch (const std::exception & error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
