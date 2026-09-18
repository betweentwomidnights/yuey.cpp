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
    std::string model_path;
    yue2::TranscriberRuntimeOptions options;
};

struct yue2_generator_context {
    std::unique_ptr<yue2::GenerationPipeline> pipeline;
    std::string model_path;
    std::string vae_path;
    std::string tokenizer_path;
    yue2::GenerationPipelineOptions options;
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

yue2::Transcriber & ensure_transcriber(yue2_transcriber_context * context) {
    if (!context->transcriber) {
        context->transcriber = std::make_unique<yue2::Transcriber>(
            context->model_path, context->options);
    }
    return *context->transcriber;
}

yue2::GenerationPipeline & ensure_generator(yue2_generator_context * context) {
    if (!context->pipeline) {
        context->pipeline = std::make_unique<yue2::GenerationPipeline>(
            context->model_path, context->vae_path, context->tokenizer_path,
            context->options);
    }
    return *context->pipeline;
}

bool initialize(
    yue2_transcription_result * result,
    char * error,
    int32_t error_size) {
    if (!result || result->size < sizeof(*result)) {
        set_error(error, error_size, "transcription result is null or too small");
        return false;
    }
    const auto caller_size = result->size;
    std::memset(result, 0, sizeof(*result));
    result->size = caller_size;
    return true;
}

bool initialize(
    yue2_generation_result * result,
    char * error,
    int32_t error_size) {
    constexpr std::size_t original_size = offsetof(yue2_generation_result, midi);
    if (!result || result->size < original_size) {
        set_error(error, error_size, "generation result is null or too small");
        return false;
    }
    const auto caller_size = result->size;
    std::memset(result, 0, std::min<std::size_t>(caller_size, sizeof(*result)));
    result->size = caller_size;
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
        context->model_path = config->model_path;
        context->options = options;
        context->transcriber = std::make_unique<yue2::Transcriber>(context->model_path, options);
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
        const auto native = ensure_transcriber(context).transcribe_mono(
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
    if (!result || result->size < sizeof(std::uint32_t)) return;
    const auto caller_size = result->size;
    if (caller_size >= offsetof(yue2_transcription_result, abc) + sizeof(result->abc)) {
        std::free(result->abc);
    }
    if (caller_size >= offsetof(yue2_transcription_result, midi) + sizeof(result->midi)) {
        std::free(result->midi);
    }
    if (caller_size >= offsetof(yue2_transcription_result, events_json) +
        sizeof(result->events_json)) {
        std::free(result->events_json);
    }
    std::memset(result, 0, std::min<std::size_t>(caller_size, sizeof(*result)));
    result->size = caller_size;
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
        context->model_path = config->model_path;
        context->vae_path = config->vae_path;
        context->tokenizer_path = config->tokenizer_path;
        context->options = options;
        context->pipeline = std::make_unique<yue2::GenerationPipeline>(
            context->model_path, context->vae_path, context->tokenizer_path, options);
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
        const auto instrumental_size = offsetof(yue2_generation_request, instrumental) +
            sizeof(request->instrumental);
        if (request->size >= instrumental_size) {
            song.instrumental = request->instrumental != 0;
        }
        const auto target_bars_size = offsetof(yue2_generation_request, target_bars) +
            sizeof(request->target_bars);
        if (request->size >= target_bars_size) song.target_bars = request->target_bars;
        const auto ending_mode_size = offsetof(yue2_generation_request, ending_mode) +
            sizeof(request->ending_mode);
        if (request->size >= ending_mode_size) {
            if (request->ending_mode == YUE2_ENDING_NATURAL) {
                song.ending_mode = yue2::EndingMode::natural;
            } else if (request->ending_mode == YUE2_ENDING_OUTRO) {
                song.ending_mode = yue2::EndingMode::outro;
            } else {
                throw std::invalid_argument("invalid ending mode");
            }
        }
        const auto outro_bars_size = offsetof(yue2_generation_request, outro_bars) +
            sizeof(request->outro_bars);
        if (request->size >= outro_bars_size && request->outro_bars != 0) {
            song.outro_bars = request->outro_bars;
        }
        const auto semantic_prefix_count_size =
            offsetof(yue2_generation_request, semantic_prefix_count) +
            sizeof(request->semantic_prefix_count);
        if (request->size >= semantic_prefix_count_size) {
            if (request->semantic_prefix_count != 0 && !request->semantic_prefix) {
                throw std::invalid_argument(
                    "semantic prefix pointer is null for a nonzero count");
            }
            if (request->semantic_prefix_count >
                static_cast<std::uint64_t>(std::numeric_limits<std::size_t>::max())) {
                throw std::invalid_argument("semantic prefix is too large");
            }
            if (request->semantic_prefix_count != 0) {
                song.semantic_prefix.assign(
                    request->semantic_prefix,
                    request->semantic_prefix +
                        static_cast<std::size_t>(request->semantic_prefix_count));
            }
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
            options.semantic_budget_explicit = true;
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
        const auto native = ensure_generator(context).generate(song, options, control);
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
#define YUE2_COPY_LEGACY_MIDI(field, source) \
        if (result->size >= offsetof(yue2_generation_result, field##_size) + \
            sizeof(result->field##_size)) { \
            result->field = copy_vector(native.midi_exports.source); \
            result->field##_size = native.midi_exports.source.size(); \
        }
        YUE2_COPY_LEGACY_MIDI(midi, transcription);
        YUE2_COPY_LEGACY_MIDI(melody_midi, melody);
        YUE2_COPY_LEGACY_MIDI(vocal_midi, vocal);
        YUE2_COPY_LEGACY_MIDI(instrumental_midi, instrumental);
        YUE2_COPY_LEGACY_MIDI(chords_midi, chords);
#undef YUE2_COPY_LEGACY_MIDI
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
    if (!result || result->size < sizeof(std::uint32_t)) return;
    const auto caller_size = result->size;
    if (caller_size >= offsetof(yue2_generation_result, interleaved_samples) +
        sizeof(result->interleaved_samples)) {
        std::free(result->interleaved_samples);
    }
    if (caller_size >= offsetof(yue2_generation_result, abc) + sizeof(result->abc)) {
        std::free(result->abc);
    }
    if (caller_size >= offsetof(yue2_generation_result, semantic_codec_ids) +
        sizeof(result->semantic_codec_ids)) {
        std::free(result->semantic_codec_ids);
    }
    if (caller_size >= offsetof(yue2_generation_result, latents) + sizeof(result->latents)) {
        std::free(result->latents);
    }
#define YUE2_FREE_LEGACY_MIDI(field) \
    if (caller_size >= offsetof(yue2_generation_result, field) + \
        sizeof(result->field)) std::free(result->field)
    YUE2_FREE_LEGACY_MIDI(midi);
    YUE2_FREE_LEGACY_MIDI(melody_midi);
    YUE2_FREE_LEGACY_MIDI(vocal_midi);
    YUE2_FREE_LEGACY_MIDI(instrumental_midi);
    YUE2_FREE_LEGACY_MIDI(chords_midi);
#undef YUE2_FREE_LEGACY_MIDI
    std::memset(result, 0, std::min<std::size_t>(caller_size, sizeof(*result)));
    result->size = caller_size;
}

YUE2_API void yue2_generator_free(yue2_generator_context * context) {
    delete context;
}

YUE2_API const char * yue2_c_version(void) {
    return yue2::version();
}

} // extern "C"

namespace {

template <typename T>
void init_v1_struct(T * value) {
    if (!value || value->size < sizeof(std::uint32_t)) return;
    const auto caller_size = value->size;
    std::memset(value, 0, std::min<std::size_t>(caller_size, sizeof(T)));
    value->size = caller_size;
}

template <typename T>
bool has_v1_size(const T * value, std::uint32_t minimum) {
    return value && value->size >= minimum;
}

void YUE2_CALL v1_error_init(yue2_error_v1 * error) {
    init_v1_struct(error);
}

void set_v1_error_literal(
    yue2_error_v1 * error, yue2_status_v1 code, const char * message) {
    if (!error || error->size < sizeof(std::uint32_t)) return;
    const auto caller_size = error->size;
    std::memset(error, 0, std::min<std::size_t>(caller_size, sizeof(*error)));
    error->size = caller_size;
    if (caller_size >= offsetof(yue2_error_v1, code) + sizeof(error->code)) {
        error->code = code;
    }
    if (caller_size > offsetof(yue2_error_v1, message)) {
        const auto capacity = std::min<std::size_t>(
            sizeof(error->message), caller_size - offsetof(yue2_error_v1, message));
        if (capacity && message) {
            std::strncpy(error->message, message, capacity - 1);
            error->message[capacity - 1] = '\0';
        }
    }
}

void set_v1_error(
    yue2_error_v1 * error, yue2_status_v1 code, const std::string & message) {
    set_v1_error_literal(error, code, message.c_str());
}

yue2_status_v1 fail_v1(
    yue2_error_v1 * error, yue2_status_v1 status, const std::string & message) {
    set_v1_error(error, status, message);
    return status;
}

yue2_status_v1 fail_v1_literal(
    yue2_error_v1 * error, yue2_status_v1 status, const char * message) {
    set_v1_error_literal(error, status, message);
    return status;
}

void clear_v1_error(yue2_error_v1 * error) {
    set_v1_error_literal(error, YUE2_STATUS_OK_V1, "");
}

const char * YUE2_CALL v1_runtime_version(void) {
    return yue2::version();
}

void YUE2_CALL v1_audio_view_init(yue2_audio_view_v1 * audio) {
    init_v1_struct(audio);
    if (audio && audio->size >= YUE2_AUDIO_VIEW_V1_MIN_SIZE) {
        audio->layout = YUE2_AUDIO_PLANAR_V1;
    }
}

void YUE2_CALL v1_transcriber_config_init(yue2_transcriber_config_v1 * config) {
    init_v1_struct(config);
}

void YUE2_CALL v1_transcription_request_init(yue2_transcription_request_v1 * request) {
    init_v1_struct(request);
    if (!request || request->size < YUE2_TRANSCRIPTION_REQUEST_V1_MIN_SIZE) return;
    request->audio.size = sizeof(request->audio);
    request->audio.layout = YUE2_AUDIO_PLANAR_V1;
    request->melody_only = 1;
    request->preset = YUE2_TRANSCRIPTION_STANDARD_V1;
    request->window_seconds = 300.0F;
    request->overlap_seconds = 200.0F;
    request->lookahead_seconds = 100.0F;
    request->max_tokens = 5120;
}

void YUE2_CALL v1_transcription_result_init(yue2_transcription_result_v1 * result) {
    init_v1_struct(result);
}

void YUE2_CALL v1_adapter_init(yue2_adapter_v1 * adapter) {
    init_v1_struct(adapter);
    if (adapter && adapter->size >= YUE2_ADAPTER_V1_MIN_SIZE) adapter->strength = 1.0F;
}

void YUE2_CALL v1_generator_config_init(yue2_generator_config_v1 * config) {
    init_v1_struct(config);
    if (config && config->size >= YUE2_GENERATOR_CONFIG_V1_MIN_SIZE) {
        config->adapter_stride = sizeof(yue2_adapter_v1);
    }
}

void YUE2_CALL v1_generation_request_init(yue2_generation_request_v1 * request) {
    init_v1_struct(request);
    if (!request || request->size < YUE2_GENERATION_REQUEST_V1_MIN_SIZE) return;
    request->symbolic_mode = YUE2_SYMBOLIC_FULL_V1;
    request->seed = 831001;
    request->abc_min_tokens = 32;
    request->abc_max_tokens = 4096;
    request->abc_temperature = 0.7F;
    request->abc_top_k = 30;
    request->abc_top_p = 0.9F;
    request->abc_repetition_penalty = 1.005F;
    request->abc_penalty_window = 100;
    request->semantic_min_tokens = 200;
    request->semantic_max_tokens = 9000;
    request->semantic_temperature = 1.0F;
    request->semantic_top_k = 100;
    request->semantic_top_p = 0.95F;
    request->semantic_repetition_penalty = 1.2F;
    request->semantic_penalty_window = 50;
    request->ode_steps = 32;
    request->ending_mode = YUE2_ENDING_NATURAL_V1;
    request->outro_bars = 4;
}

void YUE2_CALL v1_plan_result_init(yue2_plan_result_v1 * result) {
    init_v1_struct(result);
}

void YUE2_CALL v1_generation_result_init(yue2_generation_result_v1 * result) {
    init_v1_struct(result);
}

bool cancelled_message(const char * message) {
    return message && (std::strstr(message, "cancelled") || std::strstr(message, "canceled"));
}

yue2_status_v1 exception_status(const std::exception & exception) {
    if (dynamic_cast<const std::invalid_argument *>(&exception)) {
        return YUE2_STATUS_INVALID_ARGUMENT_V1;
    }
    if (cancelled_message(exception.what())) return YUE2_STATUS_CANCELLED_V1;
    return YUE2_STATUS_MODEL_ERROR_V1;
}

void emit_v1_progress(
    yue2_progress_callback_v1 callback,
    void * user,
    yue2_progress_stage_v1 stage,
    const char * stage_name_value,
    std::uint32_t current,
    std::uint32_t total) {
    if (!callback) return;
    yue2_progress_v1 progress{};
    progress.size = sizeof(progress);
    progress.stage = stage;
    progress.stage_name = stage_name_value;
    progress.current = current;
    progress.total = total;
    progress.fraction = total
        ? static_cast<float>(current) / static_cast<float>(total) : 0.0F;
    callback(user, &progress);
}

yue2_progress_stage_v1 v1_generation_stage(yue2::GenerationStage stage) {
    switch (stage) {
        case yue2::GenerationStage::abc: return YUE2_PROGRESS_ABC_V1;
        case yue2::GenerationStage::semantic: return YUE2_PROGRESS_SEMANTIC_V1;
        case yue2::GenerationStage::flow: return YUE2_PROGRESS_FLOW_V1;
        case yue2::GenerationStage::decode: return YUE2_PROGRESS_DECODE_V1;
        case yue2::GenerationStage::complete: return YUE2_PROGRESS_DONE_V1;
    }
    return YUE2_PROGRESS_OTHER_V1;
}

yue2::SymbolicMode v1_symbolic_mode(yue2_symbolic_mode_v1 value) {
    switch (value) {
        case YUE2_SYMBOLIC_FULL_V1: return yue2::SymbolicMode::full;
        case YUE2_SYMBOLIC_MELODY_V1: return yue2::SymbolicMode::melody;
        case YUE2_SYMBOLIC_OFF_V1: return yue2::SymbolicMode::off;
        default: throw std::invalid_argument("invalid symbolic mode");
    }
}

yue2::EndingMode v1_ending_mode(yue2_ending_mode_v1 value) {
    switch (value) {
        case YUE2_ENDING_NATURAL_V1: return yue2::EndingMode::natural;
        case YUE2_ENDING_OUTRO_V1: return yue2::EndingMode::outro;
        default: throw std::invalid_argument("invalid ending mode");
    }
}

yue2::SongRequest make_v1_song_request(const yue2_generation_request_v1 & request) {
    yue2::SongRequest song;
    if (request.style) song.style = request.style;
    if (request.lyrics) song.lyrics = request.lyrics;
    if (request.abc) song.abc = request.abc;
    if (request.abc_prefix) song.abc_prefix = request.abc_prefix;
    song.symbolic_mode = v1_symbolic_mode(request.symbolic_mode);
    song.seed = request.seed;
    if (request.guidance_set) song.guidance_scale = request.guidance_scale;
    song.experimental_vocal_rest = request.experimental_vocal_rest != 0;
    song.instrumental = request.instrumental != 0;
    song.target_bars = request.target_bars;
    song.ending_mode = v1_ending_mode(request.ending_mode);
    song.outro_bars = request.outro_bars;
    return song;
}

yue2::GenerationRunOptions make_v1_run_options(
    const yue2_generation_request_v1 & request) {
    yue2::GenerationRunOptions options;
    auto & abc = options.generation.abc;
    abc.min_tokens = request.abc_min_tokens;
    abc.max_tokens = request.abc_max_tokens;
    abc.temperature = request.abc_temperature;
    abc.top_k = request.abc_top_k;
    abc.top_p = request.abc_top_p;
    abc.repetition_penalty = request.abc_repetition_penalty;
    abc.penalty_window = request.abc_penalty_window;
    auto & semantic = options.generation.semantic;
    semantic.min_tokens = request.semantic_min_tokens;
    semantic.max_tokens = request.semantic_max_tokens;
    semantic.temperature = request.semantic_temperature;
    semantic.top_k = request.semantic_top_k;
    semantic.top_p = request.semantic_top_p;
    semantic.repetition_penalty = request.semantic_repetition_penalty;
    semantic.penalty_window = request.semantic_penalty_window;
    options.semantic_budget_explicit = request.semantic_budget_explicit != 0;
    options.flow.ode_steps = request.ode_steps;
    return options;
}

yue2::GenerationControl make_v1_generation_control(
    const yue2_generation_request_v1 * request) {
    yue2::GenerationControl control;
    if (request->on_progress) {
        control.on_progress = [request](
            yue2::GenerationStage stage, std::uint32_t current, std::uint32_t total) {
            emit_v1_progress(
                request->on_progress, request->callback_user,
                v1_generation_stage(stage), stage_name(stage), current, total);
        };
    }
    if (request->should_cancel) {
        control.should_cancel = [request]() {
            return request->should_cancel(request->callback_user) != 0;
        };
    }
    return control;
}

bool valid_generation_request(const yue2_generation_request_v1 * request) {
    return has_v1_size(request, YUE2_GENERATION_REQUEST_V1_MIN_SIZE) &&
        request->abc_max_tokens >= request->abc_min_tokens &&
        request->semantic_max_tokens >= request->semantic_min_tokens &&
        request->outro_bars > 0 && request->ode_steps > 0 &&
        std::isfinite(request->abc_temperature) &&
        std::isfinite(request->abc_top_p) &&
        std::isfinite(request->abc_repetition_penalty) &&
        std::isfinite(request->semantic_temperature) &&
        std::isfinite(request->semantic_top_p) &&
        std::isfinite(request->semantic_repetition_penalty) &&
        (!request->guidance_set || std::isfinite(request->guidance_scale));
}

yue2_status_v1 YUE2_CALL v1_transcriber_create(
    const yue2_transcriber_config_v1 * config,
    yue2_transcriber_context ** out_context,
    yue2_error_v1 * error) {
    clear_v1_error(error);
    if (out_context) *out_context = nullptr;
    if (!has_v1_size(config, YUE2_TRANSCRIBER_CONFIG_V1_MIN_SIZE) ||
        !out_context || !config->model_path || !*config->model_path || config->threads < 0) {
        return fail_v1(error, YUE2_STATUS_INVALID_ARGUMENT_V1,
                       "invalid transcriber configuration");
    }
    try {
        auto context = std::make_unique<yue2_transcriber_context>();
        context->model_path = config->model_path;
        if (config->device) context->options.device = config->device;
        context->options.threads = config->threads;
        context->transcriber = std::make_unique<yue2::Transcriber>(
            context->model_path, context->options);
        *out_context = context.release();
        return YUE2_STATUS_OK_V1;
    } catch (const std::bad_alloc &) {
        return fail_v1_literal(error, YUE2_STATUS_OUT_OF_MEMORY_V1,
                               "out of memory creating transcriber");
    } catch (const std::exception & exception) {
        return fail_v1_literal(error, YUE2_STATUS_MODEL_ERROR_V1, exception.what());
    } catch (...) {
        return fail_v1_literal(error, YUE2_STATUS_INTERNAL_ERROR_V1,
                               "unknown error creating transcriber");
    }
}

void YUE2_CALL v1_transcriber_unload(yue2_transcriber_context * context) {
    if (context) context->transcriber.reset();
}

void YUE2_CALL v1_transcriber_destroy(yue2_transcriber_context * context) {
    delete context;
}

void YUE2_CALL v1_transcription_result_free(yue2_transcription_result_v1 * result) {
    if (!result || result->size < sizeof(std::uint32_t)) return;
    const auto caller_size = result->size;
#define YUE2_FREE_V1_FIELD(type, field) \
    if (caller_size >= offsetof(type, field) + sizeof(result->field)) std::free(result->field)
    YUE2_FREE_V1_FIELD(yue2_transcription_result_v1, abc);
    YUE2_FREE_V1_FIELD(yue2_transcription_result_v1, midi);
    YUE2_FREE_V1_FIELD(yue2_transcription_result_v1, melody_midi);
    YUE2_FREE_V1_FIELD(yue2_transcription_result_v1, vocal_midi);
    YUE2_FREE_V1_FIELD(yue2_transcription_result_v1, instrumental_midi);
    YUE2_FREE_V1_FIELD(yue2_transcription_result_v1, chords_midi);
    YUE2_FREE_V1_FIELD(yue2_transcription_result_v1, events_json);
#undef YUE2_FREE_V1_FIELD
    std::memset(result, 0, std::min<std::size_t>(caller_size, sizeof(*result)));
    result->size = caller_size;
}

yue2_status_v1 YUE2_CALL v1_transcribe(
    yue2_transcriber_context * context,
    const yue2_transcription_request_v1 * request,
    yue2_transcription_result_v1 * result,
    yue2_error_v1 * error) {
    clear_v1_error(error);
    if (!context || !has_v1_size(request, YUE2_TRANSCRIPTION_REQUEST_V1_MIN_SIZE) ||
        !has_v1_size(result, YUE2_TRANSCRIPTION_RESULT_V1_MIN_SIZE) ||
        request->audio.size < YUE2_AUDIO_VIEW_V1_MIN_SIZE ||
        !request->audio.samples || !request->audio.frame_count ||
        !request->audio.channels || !request->audio.sample_rate ||
        request->audio.sample_rate >
            static_cast<std::uint32_t>(std::numeric_limits<std::int32_t>::max()) ||
        request->max_tokens > std::numeric_limits<std::size_t>::max()) {
        return fail_v1(error, YUE2_STATUS_INVALID_ARGUMENT_V1,
                       "invalid transcription request or result");
    }
    if (result->abc || result->midi || result->melody_midi || result->vocal_midi ||
        result->instrumental_midi || result->chords_midi || result->events_json) {
        return fail_v1(error, YUE2_STATUS_INVALID_ARGUMENT_V1,
                       "transcription result still owns data");
    }
    const auto channels = static_cast<std::size_t>(request->audio.channels);
    const auto frames = static_cast<std::size_t>(request->audio.frame_count);
    if (request->audio.frame_count > std::numeric_limits<std::size_t>::max() ||
        frames > std::numeric_limits<std::size_t>::max() / channels ||
        (request->audio.layout != YUE2_AUDIO_PLANAR_V1 &&
         request->audio.layout != YUE2_AUDIO_INTERLEAVED_V1)) {
        return fail_v1(error, YUE2_STATUS_INVALID_ARGUMENT_V1,
                       "invalid transcription audio view");
    }
    try {
        std::vector<float> mono(frames);
        for (std::size_t frame = 0; frame < frames; ++frame) {
            double sum = 0.0;
            for (std::size_t channel = 0; channel < channels; ++channel) {
                const auto index = request->audio.layout == YUE2_AUDIO_PLANAR_V1
                    ? channel * frames + frame : frame * channels + channel;
                const auto sample = request->audio.samples[index];
                if (!std::isfinite(sample)) {
                    throw std::invalid_argument("transcription input contains a non-finite sample");
                }
                sum += sample;
            }
            mono[frame] = static_cast<float>(sum / channels);
        }
        yue2::TranscriptionOptions options;
        options.melody_only = request->melody_only != 0;
        if (request->preset == YUE2_TRANSCRIPTION_PAPER_V1) {
            options.preset = yue2::TranscriptionPreset::paper;
        } else if (request->preset != YUE2_TRANSCRIPTION_STANDARD_V1) {
            throw std::invalid_argument("invalid transcription preset");
        }
        options.window_seconds = request->window_seconds;
        options.overlap_seconds = request->overlap_seconds;
        options.lookahead_seconds = request->lookahead_seconds;
        options.max_tokens = static_cast<std::size_t>(request->max_tokens);
        yue2::TranscriptionControl control;
        if (request->on_progress) {
            control.on_progress = [request](std::size_t current, std::size_t total) {
                emit_v1_progress(
                    request->on_progress, request->callback_user,
                    YUE2_PROGRESS_TRANSCRIPTION_V1, "transcription",
                    static_cast<std::uint32_t>(current),
                    static_cast<std::uint32_t>(total));
            };
        }
        if (request->should_cancel) {
            control.should_cancel = [request]() {
                return request->should_cancel(request->callback_user) != 0;
            };
        }
        const auto native = ensure_transcriber(context).transcribe_mono(
            mono.data(), mono.size(), static_cast<std::int32_t>(request->audio.sample_rate),
            options, control);
        result->abc = copy_string(native.abc);
        result->midi = copy_vector(native.midi_exports.transcription);
        result->midi_size = native.midi_exports.transcription.size();
        result->melody_midi = copy_vector(native.midi_exports.melody);
        result->melody_midi_size = native.midi_exports.melody.size();
        result->vocal_midi = copy_vector(native.midi_exports.vocal);
        result->vocal_midi_size = native.midi_exports.vocal.size();
        result->instrumental_midi = copy_vector(native.midi_exports.instrumental);
        result->instrumental_midi_size = native.midi_exports.instrumental.size();
        result->chords_midi = copy_vector(native.midi_exports.chords);
        result->chords_midi_size = native.midi_exports.chords.size();
        result->events_json = copy_string(yue2::serialize_transcription_json(native));
        result->event_count = native.events.size();
        result->duration_seconds = native.duration_seconds;
        return YUE2_STATUS_OK_V1;
    } catch (const std::bad_alloc &) {
        v1_transcription_result_free(result);
        return fail_v1_literal(error, YUE2_STATUS_OUT_OF_MEMORY_V1,
                               "out of memory during transcription");
    } catch (const std::exception & exception) {
        v1_transcription_result_free(result);
        return fail_v1_literal(error, exception_status(exception), exception.what());
    } catch (...) {
        v1_transcription_result_free(result);
        return fail_v1_literal(error, YUE2_STATUS_INTERNAL_ERROR_V1,
                               "unknown transcription error");
    }
}

yue2_status_v1 YUE2_CALL v1_generator_create(
    const yue2_generator_config_v1 * config,
    yue2_generator_context ** out_context,
    yue2_error_v1 * error) {
    clear_v1_error(error);
    if (out_context) *out_context = nullptr;
    if (!has_v1_size(config, YUE2_GENERATOR_CONFIG_V1_MIN_SIZE) || !out_context ||
        !config->model_path || !*config->model_path || !config->vae_path ||
        !*config->vae_path || !config->tokenizer_path || !*config->tokenizer_path ||
        config->threads < 0 || (config->adapter_count && !config->adapters) ||
        (config->adapter_count && config->adapter_stride < YUE2_ADAPTER_V1_MIN_SIZE)) {
        return fail_v1(error, YUE2_STATUS_INVALID_ARGUMENT_V1,
                       "invalid generator configuration");
    }
    try {
        auto context = std::make_unique<yue2_generator_context>();
        context->model_path = config->model_path;
        context->vae_path = config->vae_path;
        context->tokenizer_path = config->tokenizer_path;
        if (config->device) context->options.autoregressive.device = config->device;
        context->options.autoregressive.threads = config->threads;
        const auto * bytes = reinterpret_cast<const unsigned char *>(config->adapters);
        for (std::uint32_t index = 0; index < config->adapter_count; ++index) {
            const auto * adapter = reinterpret_cast<const yue2_adapter_v1 *>(
                bytes + static_cast<std::size_t>(index) * config->adapter_stride);
            if (!has_v1_size(adapter, YUE2_ADAPTER_V1_MIN_SIZE) ||
                !adapter->path || !*adapter->path || !std::isfinite(adapter->strength)) {
                throw std::invalid_argument("invalid LoRA adapter configuration");
            }
            context->options.autoregressive.lora_adapters.push_back(
                {adapter->path, adapter->strength});
        }
        context->pipeline = std::make_unique<yue2::GenerationPipeline>(
            context->model_path, context->vae_path, context->tokenizer_path,
            context->options);
        *out_context = context.release();
        return YUE2_STATUS_OK_V1;
    } catch (const std::bad_alloc &) {
        return fail_v1_literal(error, YUE2_STATUS_OUT_OF_MEMORY_V1,
                               "out of memory creating generator");
    } catch (const std::invalid_argument & exception) {
        return fail_v1_literal(error, YUE2_STATUS_INVALID_ARGUMENT_V1, exception.what());
    } catch (const std::exception & exception) {
        return fail_v1_literal(error, YUE2_STATUS_MODEL_ERROR_V1, exception.what());
    } catch (...) {
        return fail_v1_literal(error, YUE2_STATUS_INTERNAL_ERROR_V1,
                               "unknown error creating generator");
    }
}

void YUE2_CALL v1_generator_unload(yue2_generator_context * context) {
    if (context) context->pipeline.reset();
}

void YUE2_CALL v1_generator_destroy(yue2_generator_context * context) {
    delete context;
}

void YUE2_CALL v1_plan_result_free(yue2_plan_result_v1 * result) {
    if (!result || result->size < sizeof(std::uint32_t)) return;
    const auto caller_size = result->size;
    if (caller_size >= offsetof(yue2_plan_result_v1, abc) + sizeof(result->abc)) {
        std::free(result->abc);
    }
    if (caller_size >= offsetof(yue2_plan_result_v1, abc_token_ids) +
        sizeof(result->abc_token_ids)) {
        std::free(result->abc_token_ids);
    }
#define YUE2_FREE_PLAN_MIDI_V1(field) \
    if (caller_size >= offsetof(yue2_plan_result_v1, field) + \
        sizeof(result->field)) std::free(result->field)
    YUE2_FREE_PLAN_MIDI_V1(midi);
    YUE2_FREE_PLAN_MIDI_V1(melody_midi);
    YUE2_FREE_PLAN_MIDI_V1(vocal_midi);
    YUE2_FREE_PLAN_MIDI_V1(instrumental_midi);
    YUE2_FREE_PLAN_MIDI_V1(chords_midi);
#undef YUE2_FREE_PLAN_MIDI_V1
    std::memset(result, 0, std::min<std::size_t>(caller_size, sizeof(*result)));
    result->size = caller_size;
}

yue2_status_v1 YUE2_CALL v1_plan(
    yue2_generator_context * context,
    const yue2_generation_request_v1 * request,
    yue2_plan_result_v1 * result,
    yue2_error_v1 * error) {
    clear_v1_error(error);
    if (!context || !valid_generation_request(request) ||
        !has_v1_size(result, YUE2_PLAN_RESULT_V1_MIN_SIZE)) {
        return fail_v1(error, YUE2_STATUS_INVALID_ARGUMENT_V1,
                       "invalid planning request or result");
    }
    if (result->abc || result->abc_token_ids ||
        (result->size >= offsetof(yue2_plan_result_v1, midi) + sizeof(result->midi) && result->midi) ||
        (result->size >= offsetof(yue2_plan_result_v1, melody_midi) + sizeof(result->melody_midi) && result->melody_midi) ||
        (result->size >= offsetof(yue2_plan_result_v1, vocal_midi) + sizeof(result->vocal_midi) && result->vocal_midi) ||
        (result->size >= offsetof(yue2_plan_result_v1, instrumental_midi) + sizeof(result->instrumental_midi) && result->instrumental_midi) ||
        (result->size >= offsetof(yue2_plan_result_v1, chords_midi) + sizeof(result->chords_midi) && result->chords_midi)) {
        return fail_v1(error, YUE2_STATUS_INVALID_ARGUMENT_V1,
                       "plan result still owns data");
    }
    try {
        auto song = make_v1_song_request(*request);
        const auto native = ensure_generator(context).plan(
            song, make_v1_run_options(*request), make_v1_generation_control(request));
        result->abc = copy_string(native.abc);
        result->abc_token_ids = copy_vector(native.abc_token_ids);
        result->abc_token_count = native.abc_token_ids.size();
        result->seed = song.seed;
        result->score_bars = native.score_bars;
        result->score_duration_seconds = native.score_duration_seconds;
        result->abc_truncated = native.abc_truncated ? 1 : 0;
#define YUE2_COPY_PLAN_MIDI_V1(field, source) \
        if (result->size >= offsetof(yue2_plan_result_v1, field##_size) + \
            sizeof(result->field##_size)) { \
            result->field = copy_vector(native.midi_exports.source); \
            result->field##_size = native.midi_exports.source.size(); \
        }
        YUE2_COPY_PLAN_MIDI_V1(midi, transcription);
        YUE2_COPY_PLAN_MIDI_V1(melody_midi, melody);
        YUE2_COPY_PLAN_MIDI_V1(vocal_midi, vocal);
        YUE2_COPY_PLAN_MIDI_V1(instrumental_midi, instrumental);
        YUE2_COPY_PLAN_MIDI_V1(chords_midi, chords);
#undef YUE2_COPY_PLAN_MIDI_V1
        return YUE2_STATUS_OK_V1;
    } catch (const std::bad_alloc &) {
        v1_plan_result_free(result);
        return fail_v1_literal(error, YUE2_STATUS_OUT_OF_MEMORY_V1,
                               "out of memory during planning");
    } catch (const std::exception & exception) {
        v1_plan_result_free(result);
        return fail_v1_literal(error, exception_status(exception), exception.what());
    } catch (...) {
        v1_plan_result_free(result);
        return fail_v1_literal(error, YUE2_STATUS_INTERNAL_ERROR_V1,
                               "unknown planning error");
    }
}

void YUE2_CALL v1_generation_result_free(yue2_generation_result_v1 * result) {
    if (!result || result->size < sizeof(std::uint32_t)) return;
    const auto caller_size = result->size;
#define YUE2_FREE_GENERATION_V1_FIELD(field) \
    if (caller_size >= offsetof(yue2_generation_result_v1, field) + \
        sizeof(result->field)) std::free(result->field)
    YUE2_FREE_GENERATION_V1_FIELD(samples);
    YUE2_FREE_GENERATION_V1_FIELD(abc);
    YUE2_FREE_GENERATION_V1_FIELD(abc_token_ids);
    YUE2_FREE_GENERATION_V1_FIELD(semantic_codec_ids);
    YUE2_FREE_GENERATION_V1_FIELD(latents);
    YUE2_FREE_GENERATION_V1_FIELD(midi);
    YUE2_FREE_GENERATION_V1_FIELD(melody_midi);
    YUE2_FREE_GENERATION_V1_FIELD(vocal_midi);
    YUE2_FREE_GENERATION_V1_FIELD(instrumental_midi);
    YUE2_FREE_GENERATION_V1_FIELD(chords_midi);
#undef YUE2_FREE_GENERATION_V1_FIELD
    std::memset(result, 0, std::min<std::size_t>(caller_size, sizeof(*result)));
    result->size = caller_size;
}

yue2_status_v1 YUE2_CALL v1_generate(
    yue2_generator_context * context,
    const yue2_generation_request_v1 * request,
    yue2_generation_result_v1 * result,
    yue2_error_v1 * error) {
    clear_v1_error(error);
    if (!context || !valid_generation_request(request) ||
        !has_v1_size(result, YUE2_GENERATION_RESULT_V1_MIN_SIZE)) {
        return fail_v1(error, YUE2_STATUS_INVALID_ARGUMENT_V1,
                       "invalid generation request or result");
    }
    if (result->samples || result->abc || result->abc_token_ids ||
        result->semantic_codec_ids || result->latents ||
        (result->size >= offsetof(yue2_generation_result_v1, midi) + sizeof(result->midi) && result->midi) ||
        (result->size >= offsetof(yue2_generation_result_v1, melody_midi) + sizeof(result->melody_midi) && result->melody_midi) ||
        (result->size >= offsetof(yue2_generation_result_v1, vocal_midi) + sizeof(result->vocal_midi) && result->vocal_midi) ||
        (result->size >= offsetof(yue2_generation_result_v1, instrumental_midi) + sizeof(result->instrumental_midi) && result->instrumental_midi) ||
        (result->size >= offsetof(yue2_generation_result_v1, chords_midi) + sizeof(result->chords_midi) && result->chords_midi)) {
        return fail_v1(error, YUE2_STATUS_INVALID_ARGUMENT_V1,
                       "generation result still owns data");
    }
    try {
        auto song = make_v1_song_request(*request);
        const auto native = ensure_generator(context).generate(
            song, make_v1_run_options(*request), make_v1_generation_control(request));
        result->samples = copy_vector(native.audio.interleaved_samples);
        result->channels = static_cast<std::uint32_t>(native.audio.channels);
        result->sample_rate = static_cast<std::uint32_t>(native.audio.sample_rate);
        result->frame_count = native.audio.interleaved_samples.size() /
            static_cast<std::size_t>(native.audio.channels);
        result->abc = copy_string(native.abc);
        result->abc_token_ids = copy_vector(native.abc_token_ids);
        result->abc_token_count = native.abc_token_ids.size();
        result->semantic_codec_ids = copy_vector(native.semantic_codec_ids);
        result->semantic_count = native.semantic_codec_ids.size();
        result->latents = copy_vector(native.latents);
        result->latent_count = native.latents.size();
        result->seed = song.seed;
        result->score_bars = native.score_bars;
        result->score_duration_seconds = native.score_duration_seconds;
        result->semantic_budget = native.semantic_budget;
        result->abc_truncated = native.abc_truncated ? 1 : 0;
        result->semantic_truncated = native.semantic_truncated ? 1 : 0;
#define YUE2_COPY_GENERATION_MIDI_V1(field, source) \
        if (result->size >= offsetof(yue2_generation_result_v1, field##_size) + \
            sizeof(result->field##_size)) { \
            result->field = copy_vector(native.midi_exports.source); \
            result->field##_size = native.midi_exports.source.size(); \
        }
        YUE2_COPY_GENERATION_MIDI_V1(midi, transcription);
        YUE2_COPY_GENERATION_MIDI_V1(melody_midi, melody);
        YUE2_COPY_GENERATION_MIDI_V1(vocal_midi, vocal);
        YUE2_COPY_GENERATION_MIDI_V1(instrumental_midi, instrumental);
        YUE2_COPY_GENERATION_MIDI_V1(chords_midi, chords);
#undef YUE2_COPY_GENERATION_MIDI_V1
        return YUE2_STATUS_OK_V1;
    } catch (const std::bad_alloc &) {
        v1_generation_result_free(result);
        return fail_v1_literal(error, YUE2_STATUS_OUT_OF_MEMORY_V1,
                               "out of memory during generation");
    } catch (const std::exception & exception) {
        v1_generation_result_free(result);
        return fail_v1_literal(error, exception_status(exception), exception.what());
    } catch (...) {
        v1_generation_result_free(result);
        return fail_v1_literal(error, YUE2_STATUS_INTERNAL_ERROR_V1,
                               "unknown generation error");
    }
}

const yue2_api_v1 k_api_v1 = {
    sizeof(yue2_api_v1),
    YUE2_ABI_VERSION_1,
    v1_runtime_version,
    v1_error_init,
    v1_audio_view_init,
    v1_transcriber_config_init,
    v1_transcription_request_init,
    v1_transcription_result_init,
    v1_adapter_init,
    v1_generator_config_init,
    v1_generation_request_init,
    v1_plan_result_init,
    v1_generation_result_init,
    v1_transcriber_create,
    v1_transcriber_unload,
    v1_transcriber_destroy,
    v1_transcribe,
    v1_transcription_result_free,
    v1_generator_create,
    v1_generator_unload,
    v1_generator_destroy,
    v1_plan,
    v1_plan_result_free,
    v1_generate,
    v1_generation_result_free,
    {nullptr}
};

} // namespace

extern "C" {

YUE2_API const yue2_api_v1 * YUE2_CALL yue2_get_api(std::uint32_t abi_version) {
    return abi_version == YUE2_ABI_VERSION_1 ? &k_api_v1 : nullptr;
}

} // extern "C"
