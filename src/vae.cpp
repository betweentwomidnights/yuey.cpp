#include "yue2/vae.h"

#include "gguf_model.h"

#include "ggml-backend.h"
#include "ggml.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace yue2 {
namespace {

constexpr std::size_t graph_size = 8192;
constexpr std::int64_t latent_channels = 64;
constexpr std::array<int, 6> strides = {6, 5, 4, 4, 2, 2};
constexpr std::array<int, 6> input_channels = {2048, 1024, 512, 256, 128, 64};
constexpr std::array<int, 6> output_channels = {1024, 512, 256, 128, 64, 64};
constexpr std::array<int, 3> dilations = {1, 3, 9};

detail::GgufModel load_vae_model(
    const std::string & path,
    const VaeRuntimeOptions & options) {
    if (options.threads < 0) {
        throw std::invalid_argument("YuE2 VAE thread count cannot be negative");
    }
    return detail::load_gguf_raw(
        path.c_str(),
        options.device.empty() ? nullptr : options.device.c_str(),
        options.threads);
}

ggml_tensor * cast_f32(ggml_context * context, ggml_tensor * value) {
    return value->type == GGML_TYPE_F32 ? value : ggml_cast(context, value, GGML_TYPE_F32);
}

ggml_tensor * channel_row(ggml_context * context, ggml_tensor * value) {
    value = cast_f32(context, value);
    return ggml_reshape_2d(context, value, 1, ggml_nelements(value));
}

ggml_tensor * snake_beta(
    ggml_context * context,
    ggml_tensor * input,
    ggml_tensor * alpha_log,
    ggml_tensor * beta_log) {
    auto * alpha = ggml_exp(context, channel_row(context, alpha_log));
    auto * beta = ggml_exp(context, channel_row(context, beta_log));
    auto * sine = ggml_sin(context, ggml_mul(context, input, alpha));
    return ggml_add(context, input, ggml_div(context, ggml_sqr(context, sine), beta));
}

ggml_tensor * conv1d(
    ggml_context * context,
    ggml_tensor * input,
    ggml_tensor * weight,
    ggml_tensor * bias,
    int stride,
    int padding,
    int dilation) {
    auto * output = ggml_conv_1d(context, weight, input, stride, padding, dilation);
    output = ggml_reshape_2d(context, output, output->ne[0], output->ne[1]);
    if (bias) {
        output = ggml_add(context, output, channel_row(context, bias));
    }
    return output;
}

ggml_tensor * conv_transpose1d(
    ggml_context * context,
    ggml_tensor * input,
    ggml_tensor * native_weight,
    ggml_tensor * bias,
    int stride,
    int channels) {
    const std::int64_t kernel = native_weight->ne[0];
    const std::int64_t in_channels = native_weight->ne[2];
    auto * flattened = ggml_reshape_2d(
        context, native_weight, kernel * channels, in_channels);
    auto * col2im_weight = ggml_cont(context, ggml_transpose(context, flattened));
    auto * transposed_input = ggml_cont(context, ggml_transpose(context, input));
    auto * columns = ggml_mul_mat(context, col2im_weight, transposed_input);
    auto * output = ggml_col2im_1d(context, columns, stride, channels, (stride + 1) / 2);
    if (bias) {
        output = ggml_add(context, output, channel_row(context, bias));
    }
    return output;
}

ggml_tensor * residual_unit(
    ggml_context * context,
    detail::GgufModel & model,
    ggml_tensor * input,
    const std::string & prefix,
    int dilation) {
    auto * hidden = snake_beta(
        context,
        input,
        model.get(prefix + ".layers.0.alpha"),
        model.get(prefix + ".layers.0.beta"));
    hidden = conv1d(
        context,
        hidden,
        model.get(prefix + ".layers.1.weight"),
        model.get(prefix + ".layers.1.bias"),
        1,
        3 * dilation,
        dilation);
    hidden = snake_beta(
        context,
        hidden,
        model.get(prefix + ".layers.2.alpha"),
        model.get(prefix + ".layers.2.beta"));
    hidden = conv1d(
        context,
        hidden,
        model.get(prefix + ".layers.3.weight"),
        model.get(prefix + ".layers.3.bias"),
        1,
        0,
        1);
    return ggml_add(context, input, hidden);
}

ggml_tensor * decode_graph(
    ggml_context * context,
    detail::GgufModel & model,
    ggml_tensor * latents) {
    auto * hidden = conv1d(
        context,
        latents,
        model.get("decoder.layers.0.weight"),
        model.get("decoder.layers.0.bias"),
        1,
        3,
        1);
    for (int block = 0; block < 6; ++block) {
        const std::string prefix = "decoder.layers." + std::to_string(block + 1);
        hidden = snake_beta(
            context,
            hidden,
            model.get(prefix + ".layers.0.alpha"),
            model.get(prefix + ".layers.0.beta"));
        hidden = conv_transpose1d(
            context,
            hidden,
            model.get(prefix + ".layers.1.weight"),
            model.get(prefix + ".layers.1.bias"),
            strides[static_cast<std::size_t>(block)],
            output_channels[static_cast<std::size_t>(block)]);
        for (int residual = 0; residual < 3; ++residual) {
            hidden = residual_unit(
                context,
                model,
                hidden,
                prefix + ".layers." + std::to_string(residual + 2),
                dilations[static_cast<std::size_t>(residual)]);
        }
    }
    hidden = snake_beta(
        context,
        hidden,
        model.get("decoder.layers.7.alpha"),
        model.get("decoder.layers.7.beta"));
    return conv1d(
        context,
        hidden,
        model.get("decoder.layers.8.weight"),
        nullptr,
        1,
        3,
        1);
}

} // namespace

class VaeDecoder::Impl {
public:
    Impl(const std::string & path, const VaeRuntimeOptions & options)
        : model(load_vae_model(path, options)) {
        if (model.string("general.architecture") != "yue2_vae" ||
            model.string("yue2.component") != "vae") {
            throw std::runtime_error("[yue2:vae] not a YuE2 VAE GGUF");
        }
        if (model.u32("yue2.vae.sample_rate") != 48000 ||
            model.u32("yue2.vae.audio_channels") != 2 ||
            model.u32("yue2.vae.latent_dim") != latent_channels ||
            model.u32("yue2.vae.downsampling_ratio") != 1920 ||
            model.u32("yue2.vae.decode_core_frames") != 1024 ||
            model.u32("yue2.vae.decode_halo_frames") != 16) {
            throw std::runtime_error("[yue2:vae] unsupported VAE configuration");
        }
        tiled = options.tiled;
        core_frames = options.decode_core_frames;
        halo_frames = options.decode_halo_frames;
        if (core_frames <= 0 || halo_frames < 16) {
            throw std::invalid_argument(
                "YuE2 VAE decode tiling requires positive core frames and at least 16 halo frames");
        }
        for (std::size_t block = 0; block < strides.size(); ++block) {
            const auto & weight = *model.get(
                "decoder.layers." + std::to_string(block + 1) + ".layers.1.weight");
            if (weight.ne[2] != input_channels[block] ||
                weight.ne[1] != output_channels[block] ||
                weight.ne[0] != 2 * strides[block]) {
                throw std::runtime_error("[yue2:vae] decoder upsample tensor mismatch");
            }
        }

        ggml_backend_t backends[] = {model.backend(), nullptr};
        std::size_t backend_count = 1;
        auto * primary_device = ggml_backend_get_device(model.backend());
        if (primary_device && ggml_backend_dev_type(primary_device) != GGML_BACKEND_DEVICE_TYPE_CPU) {
            cpu_backend = detail::make_backend("cpu", options.threads, false);
            backends[backend_count++] = cpu_backend;
        }
        scheduler = ggml_backend_sched_new(
            backends, nullptr, backend_count, graph_size, false, true);
        if (!scheduler) throw std::runtime_error("[yue2:vae] could not create GGML scheduler");
    }

    ~Impl() {
        if (scheduler) ggml_backend_sched_free(scheduler);
        if (cpu_backend) ggml_backend_free(cpu_backend);
    }

    DecodedAudio decode(
        const float * values,
        std::int64_t frames,
        const VaeDecodeControl & control) {
        if (!values || frames <= 0) {
            throw std::invalid_argument("YuE2 VAE expects non-empty [frames,64] latents");
        }
        if (frames > 24576) {
            throw std::invalid_argument("YuE2 VAE latent frame count exceeds the released context");
        }
        const std::size_t value_count = static_cast<std::size_t>(frames * latent_channels);
        for (std::size_t index = 0; index < value_count; ++index) {
            if (!std::isfinite(values[index])) {
                throw std::invalid_argument("YuE2 VAE latents contain a non-finite value");
            }
        }
        std::lock_guard<std::mutex> lock(inference_mutex);

        if (control.should_cancel && control.should_cancel()) {
            throw std::runtime_error("YuE2 generation cancelled");
        }
        if (!tiled || frames <= core_frames) {
            auto result = decode_window(values, frames);
            if (control.on_progress) control.on_progress(1, 1);
            return result;
        }

        DecodedAudio result;
        const std::int64_t total_samples = frames * 1920 - 64;
        result.interleaved_samples.resize(static_cast<std::size_t>(total_samples * 2));
        std::vector<float> tile_latents;
        const auto tile_count = static_cast<std::uint32_t>(
            (frames + core_frames - 1) / core_frames);
        std::uint32_t tile_index = 0;
        for (std::int64_t start = 0; start < frames; start += core_frames) {
            if (control.should_cancel && control.should_cancel()) {
                throw std::runtime_error("YuE2 generation cancelled");
            }
            const std::int64_t end = std::min(frames, start + core_frames);
            const std::int64_t left = std::max<std::int64_t>(0, start - halo_frames);
            const std::int64_t right = std::min(frames, end + halo_frames);
            tile_latents.assign(
                values + left * latent_channels,
                values + right * latent_channels);
            const auto tile = decode_window(tile_latents.data(), right - left);
            const std::int64_t output_start = start * 1920;
            const std::int64_t output_end = std::min(end * 1920, total_samples);
            const std::int64_t copy_samples = output_end - output_start;
            const std::int64_t crop_start = (start - left) * 1920;
            if (crop_start < 0 || copy_samples <= 0 ||
                crop_start + copy_samples >
                    static_cast<std::int64_t>(tile.interleaved_samples.size() / 2)) {
                throw std::runtime_error("[yue2:vae] tile did not cover its output core");
            }
            std::copy_n(
                tile.interleaved_samples.begin() + crop_start * 2,
                static_cast<std::size_t>(copy_samples * 2),
                result.interleaved_samples.begin() + output_start * 2);
            ++tile_index;
            if (control.on_progress) control.on_progress(tile_index, tile_count);
        }
        return result;
    }

private:
    DecodedAudio decode_window(const float * values, std::int64_t frames) {
        const std::size_t value_count = static_cast<std::size_t>(frames * latent_channels);

        const std::size_t context_bytes = ggml_tensor_overhead() * graph_size +
            ggml_graph_overhead_custom(graph_size, false);
        std::vector<std::uint8_t> storage(context_bytes);
        const ggml_init_params params = {context_bytes, storage.data(), true};
        std::unique_ptr<ggml_context, decltype(&ggml_free)> context(ggml_init(params), ggml_free);
        if (!context) throw std::runtime_error("[yue2:vae] could not allocate graph context");

        auto * graph = ggml_new_graph_custom(context.get(), graph_size, false);
        // The public buffer is row-major [frames,channels]. Represent that as
        // GGML [channels,frames], then transpose into convolution's
        // [time,channels] view on the selected backend.
        auto * input = ggml_new_tensor_2d(
            context.get(), GGML_TYPE_F32, latent_channels, frames);
        ggml_set_input(input);
        ggml_set_name(input, "yue2.vae.latents");
        auto * convolution_input = ggml_cont(context.get(), ggml_transpose(context.get(), input));
        auto * planar_output = decode_graph(context.get(), model, convolution_input);
        // GGML's [samples,channels] storage is channel-planar. Transpose once
        // on the selected backend so the public buffer is ordinary interleaved
        // stereo PCM without a second host-side copy.
        auto * output = ggml_cont(context.get(), ggml_transpose(context.get(), planar_output));
        ggml_set_name(output, "yue2.vae.audio");
        ggml_build_forward_expand(graph, output);

        ggml_backend_sched_reset(scheduler);
        if (!ggml_backend_sched_alloc_graph(scheduler, graph)) {
            throw std::runtime_error("[yue2:vae] could not allocate decode graph");
        }
        ggml_backend_tensor_set(input, values, 0, value_count * sizeof(float));
        const auto status = ggml_backend_sched_graph_compute(scheduler, graph);
        if (status != GGML_STATUS_SUCCESS) {
            throw std::runtime_error(
                std::string("[yue2:vae] graph failed: ") + ggml_status_to_string(status));
        }

        const std::int64_t samples = output->ne[1];
        if (output->ne[0] != 2 || samples != frames * 1920 - 64) {
            throw std::runtime_error("[yue2:vae] decoder produced an unexpected shape");
        }
        DecodedAudio result;
        result.interleaved_samples.resize(static_cast<std::size_t>(samples * 2));
        ggml_backend_tensor_get(
            output,
            result.interleaved_samples.data(),
            0,
            result.interleaved_samples.size() * sizeof(float));
        return result;
    }

    detail::GgufModel model;
    ggml_backend_t cpu_backend = nullptr;
    ggml_backend_sched_t scheduler = nullptr;
    std::mutex inference_mutex;
    bool tiled = true;
    std::int64_t core_frames = 1024;
    std::int64_t halo_frames = 16;
};

VaeDecoder::VaeDecoder(
    const std::string & vae_gguf_path,
    const VaeRuntimeOptions & options)
    : impl_(std::make_unique<Impl>(vae_gguf_path, options)) {}

VaeDecoder::~VaeDecoder() = default;
VaeDecoder::VaeDecoder(VaeDecoder &&) noexcept = default;
VaeDecoder & VaeDecoder::operator=(VaeDecoder &&) noexcept = default;

DecodedAudio VaeDecoder::decode(
    const float * time_major_latents,
    std::int64_t latent_frames) {
    return impl_->decode(time_major_latents, latent_frames, {});
}

DecodedAudio VaeDecoder::decode(const std::vector<float> & time_major_latents) {
    if (time_major_latents.empty() || time_major_latents.size() % latent_channels != 0) {
        throw std::invalid_argument("YuE2 VAE latent vector must contain a whole number of 64-channel frames");
    }
    return impl_->decode(
        time_major_latents.data(),
        static_cast<std::int64_t>(time_major_latents.size() / latent_channels),
        {});
}

DecodedAudio VaeDecoder::decode(
    const std::vector<float> & time_major_latents,
    const VaeDecodeControl & control) {
    if (time_major_latents.empty() || time_major_latents.size() % latent_channels != 0) {
        throw std::invalid_argument("YuE2 VAE latent vector must contain a whole number of 64-channel frames");
    }
    return impl_->decode(
        time_major_latents.data(),
        static_cast<std::int64_t>(time_major_latents.size() / latent_channels),
        control);
}

} // namespace yue2
