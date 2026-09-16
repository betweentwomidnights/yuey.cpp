#include "yue2/c_api_v1.h"

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

static void YUE2_CALL on_progress(void * user, const yue2_progress_v1 * progress) {
    callbacks * value = (callbacks *)user;
    ++value->calls;
    if (progress->stage == YUE2_PROGRESS_SEMANTIC_V1) value->saw_semantic = 1;
    if (progress->stage == YUE2_PROGRESS_FLOW_V1) value->saw_flow = 1;
    if (progress->stage == YUE2_PROGRESS_DECODE_V1) value->saw_decode = 1;
    if (progress->stage == YUE2_PROGRESS_DONE_V1) value->saw_complete = 1;
    if (progress->size < YUE2_PROGRESS_V1_MIN_SIZE ||
        progress->current > progress->total || progress->fraction < 0.0f ||
        progress->fraction > 1.0f) {
        value->calls = -1000;
    }
}

static int32_t YUE2_CALL should_cancel(void * user) {
    return ((callbacks *)user)->cancel;
}

static int valid(const yue2_generation_result_v1 * result) {
    uint64_t index;
    if (!result->samples || result->frame_count == 0 || result->channels != 2 ||
        result->sample_rate != 48000 || !result->abc || !result->abc_token_ids ||
        result->abc_token_count == 0 || result->semantic_count != 2 ||
        !result->semantic_codec_ids || result->latent_count != 128 || !result->latents ||
        result->score_bars == 0 || result->score_duration_seconds <= 0.0) {
        return 0;
    }
    for (index = 0; index < result->frame_count * (uint64_t)result->channels; ++index) {
        if (!isfinite(result->samples[index])) return 0;
    }
    return 1;
}

int main(int argc, char ** argv) {
    const yue2_api_v1 * api;
    yue2_error_v1 error = { sizeof error };
    yue2_generator_config_v1 config = { sizeof config };
    yue2_generation_request_v1 request = { sizeof request };
    yue2_generation_result_v1 first = { sizeof first };
    yue2_generation_result_v1 second = { sizeof second };
    yue2_generator_context * context = NULL;
    yue2_adapter_v1 adapter = { sizeof adapter };
    yue2_status_v1 status;
    size_t pcm_bytes;
    callbacks observed;

    if (argc < 4 || argc > 6) {
        fprintf(stderr, "usage: yue2-c-generation-test MODEL.gguf VAE.gguf QWEN.TIKTOKEN [DEVICE] [LORA.gguf]\n");
        return 2;
    }
    api = yue2_get_api(YUE2_ABI_VERSION_1);
    if (!api || api->size < YUE2_API_V1_MIN_SIZE) return 1;
    api->error_init(&error);
    api->generator_config_init(&config);
    config.model_path = argv[1];
    config.vae_path = argv[2];
    config.tokenizer_path = argv[3];
    if (argc >= 5) config.device = argv[4];
    if (argc == 6) {
        api->adapter_init(&adapter);
        adapter.path = argv[5];
        config.adapters = &adapter;
        config.adapter_count = 1;
    }
    status = api->generator_create(&config, &context, &error);
    if (status != YUE2_STATUS_OK_V1) {
        fprintf(stderr, "create failed (%d): %s\n", status, error.message);
        return 1;
    }

    api->generation_request_init(&request);
    request.style = "acoustic, intimate";
    request.lyrics = "[Verse]\nA quiet line";
    request.abc =
        "X:1\nT:C ABI smoke\nM:4/4\nL:1/8\nQ:1/4=95\n"
        "V: Vocal clef=treble\nV: Ins clef=treble\nK:C\n% verse\n"
        "V: Vocal\nC8E8G16|\nV: Ins\nC,32|\n";
    request.semantic_min_tokens = 2;
    request.semantic_max_tokens = 2;
    request.semantic_budget_explicit = 1;
    request.ode_steps = 1;
    request.semantic_temperature = 0.0f;
    memset(&observed, 0, sizeof observed);
    request.on_progress = on_progress;
    request.should_cancel = should_cancel;
    request.callback_user = &observed;

    {
        yue2_plan_result_v1 plan = { sizeof plan };
        api->plan_result_init(&plan);
        status = api->plan(context, &request, &plan, &error);
        if (status != YUE2_STATUS_OK_V1 || !plan.abc || !plan.abc_token_ids ||
            plan.abc_token_count == 0 || plan.score_bars == 0 ||
            plan.score_duration_seconds <= 0.0) {
            fprintf(stderr, "planning failed (%d): %s\n", status, error.message);
            api->plan_result_free(&plan);
            api->generator_destroy(context);
            return 1;
        }
        api->plan_result_free(&plan);
    }

    api->generation_result_init(&first);
    status = api->generate(context, &request, &first, &error);
    if (status != YUE2_STATUS_OK_V1 || !valid(&first)) {
        fprintf(stderr, "first generation failed (%d): %s\n", status, error.message);
        api->generation_result_free(&first);
        api->generator_destroy(context);
        return 1;
    }
    api->generation_result_init(&second);
    status = api->generate(context, &request, &second, &error);
    pcm_bytes = (size_t)first.frame_count * (size_t)first.channels * sizeof(float);
    if (status != YUE2_STATUS_OK_V1 || !valid(&second) ||
        first.semantic_count != second.semantic_count ||
        first.latent_count != second.latent_count || first.frame_count != second.frame_count ||
        observed.calls <= 0 || !observed.saw_semantic || !observed.saw_flow ||
        !observed.saw_decode || !observed.saw_complete ||
        memcmp(first.semantic_codec_ids, second.semantic_codec_ids,
               (size_t)first.semantic_count * sizeof(int32_t)) != 0 ||
        memcmp(first.latents, second.latents,
               (size_t)first.latent_count * sizeof(float)) != 0 ||
        memcmp(first.samples, second.samples, pcm_bytes) != 0) {
        fprintf(stderr, "second generation failed or differed (%d): %s\n", status, error.message);
        api->generation_result_free(&first);
        api->generation_result_free(&second);
        api->generator_destroy(context);
        return 1;
    }

    {
        yue2_generation_result_v1 cancelled = { sizeof cancelled };
        api->generation_result_init(&cancelled);
        observed.cancel = 1;
        status = api->generate(context, &request, &cancelled, &error);
        if (status != YUE2_STATUS_CANCELLED_V1 || cancelled.samples != NULL) {
            fprintf(stderr, "cooperative cancellation failed (%d): %s\n",
                    status, error.message);
            api->generation_result_free(&cancelled);
            api->generation_result_free(&first);
            api->generation_result_free(&second);
            api->generator_destroy(context);
            return 1;
        }
        api->generation_result_free(&cancelled);
    }
    printf("C ABI V1 semantic=%d,%d latents=%llu pcm_frames=%llu\n",
           first.semantic_codec_ids[0], first.semantic_codec_ids[1],
           (unsigned long long)first.latent_count,
           (unsigned long long)first.frame_count);
    api->generation_result_free(&first);
    api->generation_result_free(&second);
    api->generator_destroy(context);
    return 0;
}
