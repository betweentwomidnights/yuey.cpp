#include "yue2/c_api.h"

#include "yue2/generation_pipeline.h"
#include "yue2/transcription.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <memory>
#include <new>
#include <stdexcept>
#include <string>
#include <vector>

struct yue2_transcriber_context {
    std::unique_ptr<yue2::Transcriber> transcriber;
};

struct yue2_generator_context {
    std::unique_ptr<yue2::GenerationPipeline> pipeline;
};

namespace {

void set_error(char * output, int32_t capacity, const std::string & message) {
    if (!output || capacity <= 0) return;
    const auto count = std::min<std::size_t>(
        message.size(), static_cast<std::size_t>(capacity - 1));
    std::memcpy(output, message.data(), count);
    output[count] = '\0';
}

char * copy_string(const std::string & value) {
    auto * output = static_cast<char *>(std::malloc(value.size() + 1));
    if (!output) throw std::bad_alloc();
    std::memcpy(output, value.data(), value.size());
    output[value.size()] = '\0';
    return output;
}

template <typename T>
T * copy_vector(const std::vector<T> & value) {
    if (value.empty()) return nullptr;
    if (value.size() > std::numeric_limits<std::size_t>::max() / sizeof(T)) {
        throw std::bad_alloc();
    }
    auto * output = static_cast<T *>(std::malloc(value.size() * sizeof(T)));
    if (!output) throw std::bad_alloc();
    std::memcpy(output, value.data(), value.size() * sizeof(T));
    return output;
}

bool initialize(
    yue2_transcription_result * result,
    char * error,
    int32_t error_size) {
    if (!result || result->size < sizeof(*result)) {
        set_error(error, error_size, "transcription result is null or too small");
        return false;
    }
    std::memset(result, 0, sizeof(*result));
    result->size = sizeof(*result);
    return true;
}

bool initialize(
    yue2_generation_result * result,
    char * error,
    int32_t error_size) {
    if (!result || result->size < sizeof(*result)) {
        set_error(error, error_size, "generation result is null or too small");
        return false;
    }
    std::memset(result, 0, sizeof(*result));
    result->size = sizeof(*result);
    return true;
}

yue2::SymbolicMode symbolic_mode(int32_t value) {
    switch (value) {
        case YUE2_SYMBOLIC_FULL: return yue2::SymbolicMode::full;
        case YUE2_SYMBOLIC_MELODY: return yue2::SymbolicMode::melody;
        case YUE2_SYMBOLIC_OFF: return yue2::SymbolicMode::off;
        default: throw std::invalid_argument("invalid symbolic mode");
    }
}

const char * stage_name(yue2::GenerationStage stage) {
    switch (stage) {
        case yue2::GenerationStage::abc: return "abc";
        case yue2::GenerationStage::semantic: return "semantic";
        case yue2::GenerationStage::flow: return "flow";
        case yue2::GenerationStage::decode: return "decode";
        case yue2::GenerationStage::complete: return "complete";
    }
    return "unknown";
}

} // namespace

extern "C" {

YUE2_API yue2_transcriber_context * yue2_transcriber_create(
    const yue2_transcriber_config * config,
    char * error,
    int32_t error_size) {
    if (!config || config->size < sizeof(*config) ||
        !config->model_path || !*config->model_path) {
        set_error(error, error_size, "invalid transcriber configuration");
        return nullptr;
    }
    try {
        yue2::TranscriberRuntimeOptions options;
        if (config->device) options.device = config->device;
        options.threads = config->threads;
        auto context = std::make_unique<yue2_transcriber_context>();
        context->transcriber = std::make_unique<yue2::Transcriber>(
            config->model_path, options);
        set_error(error, error_size, "");
        return context.release();
    } catch (const std::exception & exception) {
        set_error(error, error_size, exception.what());
        return nullptr;
    } catch (...) {
        set_error(error, error_size, "unknown transcriber initialization error");
        return nullptr;
    }
}

YUE2_API int32_t yue2_transcribe(
    yue2_transcriber_context * context,
    const yue2_transcription_request * request,
    yue2_transcription_result * result,
    char * error,
    int32_t error_size) {
    if (!initialize(result, error, error_size)) return 1;
    if (!context || !request || request->size < sizeof(*request) ||
        !request->samples || request->frame_count == 0 ||
        request->frame_count > std::numeric_limits<std::size_t>::max() ||
        request->sample_rate <= 0 || request->channels <= 0 ||
        (request->layout != YUE2_AUDIO_INTERLEAVED &&
         request->layout != YUE2_AUDIO_PLANAR)) {
        set_error(error, error_size, "invalid transcription request");
        return 1;
    }
    const auto channels = static_cast<std::size_t>(request->channels);
    const auto frames = static_cast<std::size_t>(request->frame_count);
    if (frames > std::numeric_limits<std::size_t>::max() / channels) {
        set_error(error, error_size, "transcription input is too large");
        return 1;
    }
    try {
        std::vector<float> mono(frames);
        for (std::size_t frame = 0; frame < frames; ++frame) {
            double sum = 0.0;
            for (std::size_t channel = 0; channel < channels; ++channel) {
                const auto index = request->layout == YUE2_AUDIO_PLANAR
                    ? channel * frames + frame
                    : frame * channels + channel;
                const auto sample = request->samples[index];
                if (!std::isfinite(sample)) {
                    throw std::invalid_argument("transcription input contains a non-finite sample");
                }
                sum += sample;
            }
            mono[frame] = static_cast<float>(sum / channels);
        }
        yue2::TranscriptionOptions options;
        if (request->options_set) {
            options.melody_only = request->melody_only != 0;
            if (request->preset == YUE2_TRANSCRIPTION_PAPER) {
                options.preset = yue2::TranscriptionPreset::paper;
            } else if (request->preset != YUE2_TRANSCRIPTION_STANDARD) {
                throw std::invalid_argument("invalid transcription preset");
            }
            if (request->timing_set) {
                options.window_seconds = request->window_seconds;
                options.overlap_seconds = request->overlap_seconds;
                options.lookahead_seconds = request->lookahead_seconds;
            }
            if (request->max_tokens > 0) {
                if (request->max_tokens > std::numeric_limits<std::size_t>::max()) {
                    throw std::invalid_argument("transcription token budget is too large");
                }
                options.max_tokens = static_cast<std::size_t>(request->max_tokens);
            }
        }
        yue2::TranscriptionControl control;
        if (request->on_progress) {
            control.on_progress = [request](std::size_t current, std::size_t total) {
                request->on_progress(
                    request->user, "transcription",
                    static_cast<std::uint32_t>(current),
                    static_cast<std::uint32_t>(total),
                    total ? static_cast<float>(current) / static_cast<float>(total) : 0.0F);
            };
        }
        if (request->should_cancel) {
            control.should_cancel = [request]() {
                return request->should_cancel(request->user) != 0;
            };
        }
        const auto native = context->transcriber->transcribe_mono(
            mono.data(), mono.size(), request->sample_rate, options, control);
        result->abc = copy_string(native.abc);
        result->midi = copy_vector(native.midi);
        result->midi_size = native.midi.size();
        result->events_json = copy_string(yue2::serialize_transcription_json(native));
        result->event_count = native.events.size();
        result->duration_seconds = native.duration_seconds;
        set_error(error, error_size, "");
        return 0;
    } catch (const std::exception & exception) {
        yue2_free_transcription_result(result);
        set_error(error, error_size, exception.what());
        return 2;
    } catch (...) {
        yue2_free_transcription_result(result);
        set_error(error, error_size, "unknown transcription error");
        return 2;
    }
}

YUE2_API void yue2_free_transcription_result(yue2_transcription_result * result) {
    if (!result) return;
    std::free(result->abc);
    std::free(result->midi);
    std::free(result->events_json);
    std::memset(result, 0, sizeof(*result));
    result->size = sizeof(*result);
}

YUE2_API void yue2_transcriber_free(yue2_transcriber_context * context) {
    delete context;
}

YUE2_API yue2_generator_context * yue2_generator_create(
    const yue2_generator_config * config,
    char * error,
    int32_t error_size) {
    constexpr std::size_t base_config_size =
        offsetof(yue2_generator_config, lora_adapters);
    if (!config || config->size < base_config_size ||
        !config->model_path || !*config->model_path ||
        !config->vae_path || !*config->vae_path ||
        !config->tokenizer_path || !*config->tokenizer_path) {
        set_error(error, error_size, "invalid generator configuration");
        return nullptr;
    }
    try {
        yue2::GenerationPipelineOptions options;
        if (config->device) options.autoregressive.device = config->device;
        options.autoregressive.threads = config->threads;
        const bool has_lora_fields = config->size >=
            offsetof(yue2_generator_config, lora_adapter_count) +
                sizeof(config->lora_adapter_count);
        if (has_lora_fields) {
            if (config->lora_adapter_count && !config->lora_adapters) {
                throw std::invalid_argument("LoRA adapter array is null");
            }
            options.autoregressive.lora_adapters.reserve(config->lora_adapter_count);
            for (std::uint32_t index = 0; index < config->lora_adapter_count; ++index) {
                const auto & adapter = config->lora_adapters[index];
                if (adapter.size < sizeof(adapter) || !adapter.path || !*adapter.path ||
                    !std::isfinite(adapter.strength)) {
                    throw std::invalid_argument("invalid LoRA adapter configuration");
                }
                options.autoregressive.lora_adapters.push_back(
                    {adapter.path, adapter.strength});
            }
        }
        auto context = std::make_unique<yue2_generator_context>();
        context->pipeline = std::make_unique<yue2::GenerationPipeline>(
            config->model_path, config->vae_path, config->tokenizer_path, options);
        set_error(error, error_size, "");
        return context.release();
    } catch (const std::exception & exception) {
        set_error(error, error_size, exception.what());
        return nullptr;
    } catch (...) {
        set_error(error, error_size, "unknown generator initialization error");
        return nullptr;
    }
}

YUE2_API int32_t yue2_generate(
    yue2_generator_context * context,
    const yue2_generation_request * request,
    yue2_generation_result * result,
    char * error,
    int32_t error_size) {
    if (!initialize(result, error, error_size)) return 11;
    constexpr std::size_t original_request_size =
        offsetof(yue2_generation_request, abc_prefix);
    if (!context || !request || request->size < original_request_size) {
        set_error(error, error_size, "invalid generation request");
        return 1;
    }
    try {
        yue2::SongRequest song;
        if (request->style) song.style = request->style;
        if (request->lyrics) song.lyrics = request->lyrics;
        song.symbolic_mode = symbolic_mode(request->symbolic_mode);
        if (request->abc) song.abc = request->abc;
        const auto abc_prefix_size = offsetof(yue2_generation_request, abc_prefix) +
            sizeof(request->abc_prefix);
        if (request->size >= abc_prefix_size && request->abc_prefix) {
            song.abc_prefix = request->abc_prefix;
        }
        const auto vocal_rest_size = offsetof(yue2_generation_request, experimental_vocal_rest) +
            sizeof(request->experimental_vocal_rest);
        if (request->size >= vocal_rest_size) {
            song.experimental_vocal_rest = request->experimental_vocal_rest != 0;
        }
        if (request->seed_set) song.seed = request->seed;
        if (request->guidance_set) song.guidance_scale = request->guidance_scale;

        yue2::GenerationRunOptions options;
        if (request->abc_min_tokens) options.generation.abc.min_tokens = request->abc_min_tokens;
        if (request->abc_max_tokens) options.generation.abc.max_tokens = request->abc_max_tokens;
        if (request->semantic_min_tokens) {
            options.generation.semantic.min_tokens = request->semantic_min_tokens;
        }
        if (request->semantic_max_tokens) {
            options.generation.semantic.max_tokens = request->semantic_max_tokens;
        }
        if (request->ode_steps) options.flow.ode_steps = request->ode_steps;
        if (request->temperature_set) {
            options.generation.semantic.temperature = request->temperature;
        }
        if (request->top_k) options.generation.semantic.top_k = request->top_k;
        if (request->top_p != 0.0F) options.generation.semantic.top_p = request->top_p;
        if (request->repetition_penalty != 0.0F) {
            options.generation.semantic.repetition_penalty = request->repetition_penalty;
        }
        if (request->penalty_window) {
            options.generation.semantic.penalty_window = request->penalty_window;
        }

        yue2::GenerationControl control;
        if (request->on_progress) {
            control.on_progress = [request](
                yue2::GenerationStage stage,
                std::uint32_t current,
                std::uint32_t total) {
                request->on_progress(
                    request->user, stage_name(stage), current, total,
                    total ? static_cast<float>(current) / static_cast<float>(total) : 0.0F);
            };
        }
        if (request->should_cancel) {
            control.should_cancel = [request]() {
                return request->should_cancel(request->user) != 0;
            };
        }
        const auto native = context->pipeline->generate(song, options, control);
        result->interleaved_samples = copy_vector(native.audio.interleaved_samples);
        result->channels = native.audio.channels;
        result->sample_rate = native.audio.sample_rate;
        result->frame_count = native.audio.interleaved_samples.size() /
            static_cast<std::size_t>(native.audio.channels);
        result->abc = copy_string(native.abc);
        result->semantic_codec_ids = copy_vector(native.semantic_codec_ids);
        result->semantic_count = native.semantic_codec_ids.size();
        result->latents = copy_vector(native.latents);
        result->latent_count = native.latents.size();
        result->seed = song.seed;
        result->abc_truncated = native.abc_truncated ? 1 : 0;
        result->semantic_truncated = native.semantic_truncated ? 1 : 0;
        set_error(error, error_size, "");
        return 0;
    } catch (const std::exception & exception) {
        yue2_free_generation_result(result);
        set_error(error, error_size, exception.what());
        return 2;
    } catch (...) {
        yue2_free_generation_result(result);
        set_error(error, error_size, "unknown generation error");
        return 2;
    }
}

YUE2_API void yue2_free_generation_result(yue2_generation_result * result) {
    if (!result) return;
    std::free(result->interleaved_samples);
    std::free(result->abc);
    std::free(result->semantic_codec_ids);
    std::free(result->latents);
    std::memset(result, 0, sizeof(*result));
    result->size = sizeof(*result);
}

YUE2_API void yue2_generator_free(yue2_generator_context * context) {
    delete context;
}

YUE2_API const char * yue2_c_version(void) {
    return yue2::version();
}

} // extern "C"
