#include "yue2/autoregressive.h"
#include "yue2/vae.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iostream>
#include <iterator>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

class Reader {
public:
    explicit Reader(const std::string & path) {
        std::ifstream input(path, std::ios::binary);
        if (!input) throw std::runtime_error("failed to open flow fixture: " + path);
        bytes_ = {std::istreambuf_iterator<char>(input), std::istreambuf_iterator<char>()};
    }
    void magic() {
        static constexpr std::uint8_t expected[] = {'Y', '2', 'F', 'L', 2, 0, 0, 0};
        require(sizeof(expected));
        if (std::memcmp(bytes_.data(), expected, sizeof(expected)) != 0) {
            throw std::runtime_error("invalid flow fixture header");
        }
        offset_ += sizeof(expected);
    }
    std::uint32_t u32() {
        require(4);
        std::uint32_t output = 0;
        for (int byte = 0; byte < 4; ++byte) {
            output |= static_cast<std::uint32_t>(bytes_[offset_ + byte]) << (8 * byte);
        }
        offset_ += 4;
        return output;
    }
    std::int32_t i32() { return static_cast<std::int32_t>(u32()); }
    float f32() {
        const auto bits = u32();
        float output = 0.0F;
        std::memcpy(&output, &bits, sizeof(output));
        return output;
    }
    std::vector<std::int32_t> ids() {
        std::vector<std::int32_t> output(u32());
        for (auto & value : output) value = i32();
        return output;
    }
    std::vector<float> floats() {
        std::vector<float> output(u32());
        for (auto & value : output) value = f32();
        return output;
    }
    bool done() const noexcept { return offset_ == bytes_.size(); }
private:
    void require(std::size_t count) const {
        if (count > bytes_.size() - offset_) throw std::runtime_error("truncated flow fixture");
    }
    std::vector<std::uint8_t> bytes_;
    std::size_t offset_ = 0;
};

} // namespace

int main(int argc, char ** argv) {
    if (argc < 4 || argc > 5) {
        std::cerr << "usage: yue2-flow-test MODEL.gguf VAE.gguf FIXTURE [DEVICE]\n";
        return 2;
    }
    Reader fixture(argv[3]);
    fixture.magic();
    yue2::FlowOptions flow;
    flow.ode_steps = fixture.u32();
    const auto prefix = fixture.ids();
    const auto codec = fixture.ids();
    const auto noise = fixture.floats();
    const auto expected = fixture.floats();
    const auto expected_audio = fixture.floats();
    if (!fixture.done()) throw std::runtime_error("trailing flow fixture data");

    yue2::AutoregressiveOptions options;
    if (argc == 5) options.device = argv[4];
    yue2::AutoregressiveModel model(argv[1], options);
    const auto actual = model.synthesize_latents(prefix, codec, noise, flow);
    if (actual.size() != expected.size()) throw std::runtime_error("flow result shape mismatch");
    double squared = 0.0;
    double squared_reference = 0.0;
    double mean = 0.0;
    float maximum = 0.0F;
    for (std::size_t index = 0; index < actual.size(); ++index) {
        const auto error = actual[index] - expected[index];
        maximum = std::max(maximum, std::abs(error));
        mean += std::abs(error);
        squared += static_cast<double>(error) * error;
        squared_reference += static_cast<double>(expected[index]) * expected[index];
    }
    mean /= actual.size();
    const auto rms = std::sqrt(squared / actual.size());
    const auto relative = std::sqrt(squared / squared_reference);
    std::cout << "max_error=" << maximum << " mean_error=" << mean
              << " rms_error=" << rms << " relative_rms=" << relative << "\n";
    if (maximum > 0.25F || relative > 0.05) {
        std::cerr << "YuE2 NAR flow exceeds parity tolerance\n";
        return 1;
    }
    yue2::VaeRuntimeOptions vae_options;
    vae_options.device = options.device;
    yue2::VaeDecoder decoder(argv[2], vae_options);
    const auto audio = decoder.decode(actual);
    if (audio.interleaved_samples.size() != expected_audio.size()) {
        throw std::runtime_error("flow-to-VAE audio shape mismatch");
    }
    double audio_squared = 0.0;
    double audio_mean = 0.0;
    float audio_maximum = 0.0F;
    for (std::size_t index = 0; index < expected_audio.size(); ++index) {
        const auto error = std::abs(audio.interleaved_samples[index] - expected_audio[index]);
        audio_maximum = std::max(audio_maximum, error);
        audio_mean += error;
        audio_squared += static_cast<double>(error) * error;
    }
    audio_mean /= expected_audio.size();
    const auto audio_rms = std::sqrt(audio_squared / expected_audio.size());
    std::cout << "audio_max_error=" << audio_maximum << " audio_mean_error="
              << audio_mean << " audio_rms_error=" << audio_rms << "\n";
    if (audio_maximum > 0.02F || audio_rms > 0.005) {
        std::cerr << "YuE2 flow-to-VAE audio exceeds parity tolerance\n";
        return 1;
    }
    return 0;
}
