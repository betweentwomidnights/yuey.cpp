#pragma once

#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <vector>

namespace yue2 {

struct VaeRuntimeOptions {
    std::string device;
    int threads = 0;
    bool tiled = true;
    std::int64_t decode_core_frames = 1024;
    std::int64_t decode_halo_frames = 16;
    std::int64_t encode_core_frames = 128;
    std::int64_t encode_halo_frames = 16;
};

struct DecodedAudio {
    std::vector<float> interleaved_samples;
    std::int32_t sample_rate = 48000;
    std::int32_t channels = 2;
};

struct VaeDecodeControl {
    std::function<void(std::uint32_t current, std::uint32_t total)> on_progress;
    std::function<bool()> should_cancel;
};

// Native decoder for the released YuE2 Oobleck VAE. Input is row-major
// [latent_frames, 64]; output is interleaved 48 kHz stereo float PCM.
class VaeDecoder {
public:
    explicit VaeDecoder(
        const std::string & vae_gguf_path,
        const VaeRuntimeOptions & options = {});
    ~VaeDecoder();
    VaeDecoder(VaeDecoder &&) noexcept;
    VaeDecoder & operator=(VaeDecoder &&) noexcept;
    VaeDecoder(const VaeDecoder &) = delete;
    VaeDecoder & operator=(const VaeDecoder &) = delete;

    DecodedAudio decode(
        const float * time_major_latents,
        std::int64_t latent_frames);
    DecodedAudio decode(const std::vector<float> & time_major_latents);
    DecodedAudio decode(
        const std::vector<float> & time_major_latents,
        const VaeDecodeControl & control);

private:
    class Impl;
    std::unique_ptr<Impl> impl_;
};

using VaeEncodeControl = VaeDecodeControl;

// Posterior means from 48 kHz mono/stereo PCM. Output is time-major
// [latent_frames,64], without posterior sampling. Tiling preserves the
// convolution stride phase and includes the complete receptive field.
class VaeEncoder {
public:
    explicit VaeEncoder(const std::string &vae_gguf_path, const VaeRuntimeOptions &options = {});
    ~VaeEncoder();
    VaeEncoder(VaeEncoder &&) noexcept;
    VaeEncoder &operator=(VaeEncoder &&) noexcept;
    VaeEncoder(const VaeEncoder &) = delete;
    VaeEncoder &operator=(const VaeEncoder &) = delete;
    std::vector<float> encode(const float *interleaved_pcm, std::int64_t samples_per_channel,
                              int channels = 2, const VaeEncodeControl &control = {});
private:
    class Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace yue2
