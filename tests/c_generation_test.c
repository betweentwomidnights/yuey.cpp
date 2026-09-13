#include "yue2/c_api.h"

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef struct {
    int calls;
    int saw_semantic;
    int saw_flow;
    int saw_decode;
    int saw_complete;
    int cancel;
} callbacks;

static void on_progress(
    void * user, const char * stage, uint32_t current, uint32_t total, float fraction) {
    callbacks * value = (callbacks *)user;
    ++value->calls;
    if (strcmp(stage, "semantic") == 0) value->saw_semantic = 1;
    if (strcmp(stage, "flow") == 0) value->saw_flow = 1;
    if (strcmp(stage, "decode") == 0) value->saw_decode = 1;
    if (strcmp(stage, "complete") == 0) value->saw_complete = 1;
    if (current > total || fraction < 0.0f || fraction > 1.0f) value->calls = -1000;
}

static int32_t should_cancel(void * user) {
    return ((callbacks *)user)->cancel;
}

static int valid(const yue2_generation_result * result) {
    uint64_t index;
    if (!result->interleaved_samples || result->frame_count == 0 ||
        result->channels != 2 || result->sample_rate != 48000 ||
        !result->abc || result->semantic_count != 2 ||
        !result->semantic_codec_ids || result->latent_count != 128 ||
        !result->latents) {
        return 0;
    }
    for (index = 0; index < result->frame_count * (uint64_t)result->channels; ++index) {
        if (!isfinite(result->interleaved_samples[index])) return 0;
    }
    return 1;
}

int main(int argc, char ** argv) {
    char error[512] = {0};
    yue2_generator_config config;
    yue2_generation_request request;
    yue2_generation_result first;
    yue2_generation_result second;
    yue2_generator_context * context;
    yue2_lora_adapter adapter;
    int rc;
    size_t pcm_bytes;
    callbacks observed;

    if (argc < 4 || argc > 6) {
        fprintf(stderr, "usage: yue2-c-generation-test MODEL.gguf VAE.gguf QWEN.TIKTOKEN [DEVICE] [LORA.gguf]\n");
        return 2;
    }
    memset(&config, 0, sizeof config);
    config.size = (uint32_t)sizeof config;
    config.model_path = argv[1];
    config.vae_path = argv[2];
    config.tokenizer_path = argv[3];
    if (argc >= 5) config.device = argv[4];
    if (argc == 6) {
        memset(&adapter, 0, sizeof adapter);
        adapter.size = (uint32_t)sizeof adapter;
        adapter.path = argv[5];
        adapter.strength = 1.0f;
        config.lora_adapters = &adapter;
        config.lora_adapter_count = 1;
    }
    context = yue2_generator_create(&config, error, (int32_t)sizeof error);
    if (!context) {
        fprintf(stderr, "create failed: %s\n", error);
        return 1;
    }

    memset(&request, 0, sizeof request);
    request.size = (uint32_t)sizeof request;
    request.style = "acoustic, intimate";
    request.lyrics = "[Verse]\nA quiet line";
    request.abc = "X:1\nM:4/4\nK:C\nC2 E2 G4|";
    request.symbolic_mode = YUE2_SYMBOLIC_FULL;
    request.seed = 831001;
    request.seed_set = 1;
    request.semantic_min_tokens = 2;
    request.semantic_max_tokens = 2;
    request.ode_steps = 1;
    request.temperature_set = 1;
    request.temperature = 0.0f;
    memset(&observed, 0, sizeof observed);
    request.on_progress = on_progress;
    request.should_cancel = should_cancel;
    request.user = &observed;

    memset(&first, 0, sizeof first);
    first.size = (uint32_t)sizeof first;
    rc = yue2_generate(context, &request, &first, error, (int32_t)sizeof error);
    if (rc != 0 || !valid(&first)) {
        fprintf(stderr, "first generation failed (%d): %s\n", rc, error);
        yue2_free_generation_result(&first);
        yue2_generator_free(context);
        return 1;
    }
    memset(&second, 0, sizeof second);
    second.size = (uint32_t)sizeof second;
    rc = yue2_generate(context, &request, &second, error, (int32_t)sizeof error);
    pcm_bytes = (size_t)first.frame_count * (size_t)first.channels * sizeof(float);
    if (rc != 0 || !valid(&second) || first.semantic_count != second.semantic_count ||
        first.latent_count != second.latent_count || first.frame_count != second.frame_count ||
        observed.calls <= 0 || !observed.saw_semantic || !observed.saw_flow ||
        !observed.saw_decode || !observed.saw_complete ||
        memcmp(first.semantic_codec_ids, second.semantic_codec_ids,
               (size_t)first.semantic_count * sizeof(int32_t)) != 0 ||
        memcmp(first.latents, second.latents,
               (size_t)first.latent_count * sizeof(float)) != 0 ||
        memcmp(first.interleaved_samples, second.interleaved_samples, pcm_bytes) != 0) {
        fprintf(stderr, "second generation failed or differed (%d): %s\n", rc, error);
        yue2_free_generation_result(&first);
        yue2_free_generation_result(&second);
        yue2_generator_free(context);
        return 1;
    }

    {
        yue2_generation_result cancelled;
        memset(&cancelled, 0, sizeof cancelled);
        cancelled.size = (uint32_t)sizeof cancelled;
        observed.cancel = 1;
        rc = yue2_generate(context, &request, &cancelled, error, (int32_t)sizeof error);
        if (rc == 0 || strstr(error, "cancelled") == NULL ||
            cancelled.interleaved_samples != NULL) {
            fprintf(stderr, "cooperative cancellation failed (%d): %s\n", rc, error);
            yue2_free_generation_result(&cancelled);
            yue2_free_generation_result(&first);
            yue2_free_generation_result(&second);
            yue2_generator_free(context);
            return 1;
        }
        yue2_free_generation_result(&cancelled);
    }
    printf("C ABI semantic=%d,%d latents=%llu pcm_frames=%llu\n",
           first.semantic_codec_ids[0], first.semantic_codec_ids[1],
           (unsigned long long)first.latent_count,
           (unsigned long long)first.frame_count);
    yue2_free_generation_result(&first);
    yue2_free_generation_result(&second);
    yue2_generator_free(context);
    return 0;
}
