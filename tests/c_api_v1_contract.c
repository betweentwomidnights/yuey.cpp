#include "yue2/c_api_v1.h"

#include <math.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define CHECK(condition) do {                                                     \
    if (!(condition)) {                                                           \
        fprintf(stderr, "contract check failed at %s:%d: %s\n",                \
                __FILE__, __LINE__, #condition);                                  \
        return 1;                                                                 \
    }                                                                             \
} while (0)

_Static_assert(offsetof(yue2_error_v1, size) == 0, "size must lead error");
_Static_assert(offsetof(yue2_audio_view_v1, size) == 0, "size must lead audio");
_Static_assert(offsetof(yue2_transcription_request_v1, size) == 0,
               "size must lead transcription request");
_Static_assert(offsetof(yue2_generation_request_v1, size) == 0,
               "size must lead generation request");
_Static_assert(offsetof(yue2_api_v1, size) == 0, "size must lead API table");

int main(void) {
    const yue2_api_v1 * api = yue2_get_api(YUE2_ABI_VERSION_1);
    CHECK(api != NULL);
    CHECK(yue2_get_api(0) == NULL);
    CHECK(yue2_get_api(YUE2_ABI_VERSION_1 + 1) == NULL);
    CHECK(api->size >= YUE2_API_V1_MIN_SIZE);
    CHECK(api->abi_version == YUE2_ABI_VERSION_1);
    CHECK(api->runtime_version != NULL);
    CHECK(api->runtime_version()[0] != '\0');

    yue2_error_v1 error;
    memset(&error, 0xA5, sizeof error);
    error.size = sizeof error;
    api->error_init(&error);
    CHECK(error.size == sizeof error);
    CHECK(error.code == YUE2_STATUS_OK_V1);
    CHECK(error.message[0] == '\0');

    yue2_transcriber_config_v1 transcriber_config;
    memset(&transcriber_config, 0xA5, sizeof transcriber_config);
    transcriber_config.size = sizeof transcriber_config;
    api->transcriber_config_init(&transcriber_config);
    CHECK(transcriber_config.size == sizeof transcriber_config);
    CHECK(transcriber_config.model_path == NULL);
    CHECK(transcriber_config.device == NULL);
    CHECK(transcriber_config.threads == 0);

    yue2_transcription_request_v1 transcription;
    memset(&transcription, 0xA5, sizeof transcription);
    transcription.size = sizeof transcription;
    api->transcription_request_init(&transcription);
    CHECK(transcription.size == sizeof transcription);
    CHECK(transcription.audio.size == sizeof transcription.audio);
    CHECK(transcription.audio.layout == YUE2_AUDIO_PLANAR_V1);
    CHECK(transcription.melody_only == 1);
    CHECK(transcription.preset == YUE2_TRANSCRIPTION_STANDARD_V1);
    CHECK(transcription.window_seconds == 300.0F);
    CHECK(transcription.overlap_seconds == 200.0F);
    CHECK(transcription.lookahead_seconds == 100.0F);
    CHECK(transcription.max_tokens == 5120);

    yue2_adapter_v1 adapter;
    memset(&adapter, 0xA5, sizeof adapter);
    adapter.size = sizeof adapter;
    api->adapter_init(&adapter);
    CHECK(adapter.size == sizeof adapter);
    CHECK(adapter.path == NULL);
    CHECK(adapter.strength == 1.0F);

    yue2_generator_config_v1 generator_config;
    memset(&generator_config, 0xA5, sizeof generator_config);
    generator_config.size = sizeof generator_config;
    api->generator_config_init(&generator_config);
    CHECK(generator_config.size == sizeof generator_config);
    CHECK(generator_config.model_path == NULL);
    CHECK(generator_config.adapter_stride == sizeof(yue2_adapter_v1));

    yue2_generation_request_v1 generation;
    memset(&generation, 0xA5, sizeof generation);
    generation.size = sizeof generation;
    api->generation_request_init(&generation);
    CHECK(generation.size == sizeof generation);
    CHECK(generation.symbolic_mode == YUE2_SYMBOLIC_FULL_V1);
    CHECK(generation.seed == 831001);
    CHECK(generation.abc_min_tokens == 32);
    CHECK(generation.abc_max_tokens == 4096);
    CHECK(generation.abc_temperature == 0.7F);
    CHECK(generation.semantic_min_tokens == 200);
    CHECK(generation.semantic_max_tokens == 9000);
    CHECK(generation.semantic_budget_explicit == 0);
    CHECK(generation.ode_steps == 32);
    CHECK(generation.ending_mode == YUE2_ENDING_NATURAL_V1);
    CHECK(generation.outro_bars == 4);

    /* Initializers and result frees clear only the known prefix, preserving a
     * future caller's appended storage and original size. */
    struct future_generation_request {
        yue2_generation_request_v1 known;
        uint8_t tail[19];
    } future_request;
    memset(&future_request, 0x5A, sizeof future_request);
    future_request.known.size = sizeof future_request;
    api->generation_request_init(&future_request.known);
    CHECK(future_request.known.size == sizeof future_request);
    for (size_t i = 0; i < sizeof future_request.tail; ++i) {
        CHECK(future_request.tail[i] == 0x5A);
    }

    struct future_generation_result {
        yue2_generation_result_v1 known;
        uint8_t tail[23];
    } future_result;
    memset(&future_result, 0x3C, sizeof future_result);
    future_result.known.size = sizeof future_result;
    api->generation_result_init(&future_result.known);
    future_result.known.samples = (float *)malloc(sizeof(float));
    future_result.known.abc = (char *)malloc(2);
    future_result.known.abc_token_ids = (int32_t *)malloc(sizeof(int32_t));
    future_result.known.semantic_codec_ids = (int32_t *)malloc(sizeof(int32_t));
    future_result.known.latents = (float *)malloc(sizeof(float));
    future_result.known.midi = (uint8_t *)malloc(1);
    future_result.known.melody_midi = (uint8_t *)malloc(1);
    future_result.known.vocal_midi = (uint8_t *)malloc(1);
    future_result.known.instrumental_midi = (uint8_t *)malloc(1);
    future_result.known.chords_midi = (uint8_t *)malloc(1);
    CHECK(future_result.known.samples && future_result.known.abc &&
          future_result.known.abc_token_ids && future_result.known.semantic_codec_ids &&
          future_result.known.latents && future_result.known.midi &&
          future_result.known.melody_midi && future_result.known.vocal_midi &&
          future_result.known.instrumental_midi && future_result.known.chords_midi);
    api->generation_result_free(&future_result.known);
    api->generation_result_free(&future_result.known);
    CHECK(future_result.known.size == sizeof future_result);
    CHECK(future_result.known.samples == NULL);
    CHECK(future_result.known.abc == NULL);
    CHECK(future_result.known.midi == NULL);
    CHECK(future_result.known.melody_midi == NULL);
    CHECK(future_result.known.vocal_midi == NULL);
    CHECK(future_result.known.instrumental_midi == NULL);
    CHECK(future_result.known.chords_midi == NULL);
    for (size_t i = 0; i < sizeof future_result.tail; ++i) {
        CHECK(future_result.tail[i] == 0x3C);
    }

    yue2_transcription_result_v1 transcription_result;
    memset(&transcription_result, 0xA5, sizeof transcription_result);
    transcription_result.size = sizeof transcription_result;
    api->transcription_result_init(&transcription_result);
    CHECK(transcription_result.abc == NULL);
    CHECK(transcription_result.midi == NULL);
    CHECK(transcription_result.melody_midi == NULL);
    CHECK(transcription_result.vocal_midi == NULL);
    CHECK(transcription_result.instrumental_midi == NULL);
    CHECK(transcription_result.chords_midi == NULL);
    api->transcription_result_free(&transcription_result);

    yue2_plan_result_v1 plan_result;
    memset(&plan_result, 0xA5, sizeof plan_result);
    plan_result.size = sizeof plan_result;
    api->plan_result_init(&plan_result);
    CHECK(plan_result.abc == NULL);
    CHECK(plan_result.abc_token_ids == NULL);
    CHECK(plan_result.midi == NULL);
    CHECK(plan_result.melody_midi == NULL);
    CHECK(plan_result.vocal_midi == NULL);
    CHECK(plan_result.instrumental_midi == NULL);
    CHECK(plan_result.chords_midi == NULL);
    plan_result.midi = (uint8_t *)malloc(1);
    plan_result.melody_midi = (uint8_t *)malloc(1);
    plan_result.vocal_midi = (uint8_t *)malloc(1);
    plan_result.instrumental_midi = (uint8_t *)malloc(1);
    plan_result.chords_midi = (uint8_t *)malloc(1);
    CHECK(plan_result.midi && plan_result.melody_midi && plan_result.vocal_midi &&
          plan_result.instrumental_midi && plan_result.chords_midi);
    api->plan_result_free(&plan_result);
    CHECK(plan_result.midi == NULL);
    CHECK(plan_result.melody_midi == NULL);
    CHECK(plan_result.vocal_midi == NULL);
    CHECK(plan_result.instrumental_midi == NULL);
    CHECK(plan_result.chords_midi == NULL);

    /* No weights are needed to verify typed validation and error reporting. */
    yue2_transcriber_context * transcriber = (yue2_transcriber_context *)(uintptr_t)1;
    CHECK(api->transcriber_create(&transcriber_config, &transcriber, &error) ==
          YUE2_STATUS_INVALID_ARGUMENT_V1);
    CHECK(transcriber == NULL);
    CHECK(error.code == YUE2_STATUS_INVALID_ARGUMENT_V1);
    CHECK(error.message[0] != '\0');

    yue2_generator_context * generator = (yue2_generator_context *)(uintptr_t)1;
    CHECK(api->generator_create(&generator_config, &generator, &error) ==
          YUE2_STATUS_INVALID_ARGUMENT_V1);
    CHECK(generator == NULL);
    CHECK(error.code == YUE2_STATUS_INVALID_ARGUMENT_V1);

    /* Strided future adapter entries are accepted, while too-small entries
     * are rejected before model loading. */
    struct future_adapter {
        yue2_adapter_v1 known;
        uint8_t tail[11];
    } future_adapter;
    memset(&future_adapter, 0x6D, sizeof future_adapter);
    future_adapter.known.size = sizeof future_adapter;
    api->adapter_init(&future_adapter.known);
    future_adapter.known.path = "missing-adapter.gguf";
    api->generator_config_init(&generator_config);
    generator_config.model_path = "missing-model.gguf";
    generator_config.vae_path = "missing-vae.gguf";
    generator_config.tokenizer_path = "missing-tokenizer";
    generator_config.adapters = &future_adapter.known;
    generator_config.adapter_count = 1;
    generator_config.adapter_stride = sizeof future_adapter;
    CHECK(api->generator_create(&generator_config, &generator, &error) ==
          YUE2_STATUS_MODEL_ERROR_V1);
    CHECK(generator == NULL);
    for (size_t i = 0; i < sizeof future_adapter.tail; ++i) {
        CHECK(future_adapter.tail[i] == 0x6D);
    }
    future_adapter.known.size = sizeof(uint32_t);
    CHECK(api->generator_create(&generator_config, &generator, &error) ==
          YUE2_STATUS_INVALID_ARGUMENT_V1);
    CHECK(generator == NULL);

    puts("yue2 C ABI V1 contract: ok");
    return 0;
}
