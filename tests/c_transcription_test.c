#include "yue2/c_api_v1.h"

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int progress_calls = 0;
static int cancel_now = 0;

static void YUE2_CALL on_progress(void * user, const yue2_progress_v1 * progress) {
    (void)user;
    if (progress->stage == YUE2_PROGRESS_TRANSCRIPTION_V1 &&
        progress->current <= progress->total && progress->fraction >= 0.0f &&
        progress->fraction <= 1.0f) {
        ++progress_calls;
    }
}

static int32_t YUE2_CALL should_cancel(void * user) {
    (void)user;
    return cancel_now;
}

static int midi_file(const uint8_t * data, uint64_t size) {
    return data && size >= 22 && memcmp(data, "MThd", 4) == 0;
}

int main(int argc, char ** argv) {
    enum { frames = 4800, channels = 2, sample_rate = 24000 };
    const yue2_api_v1 * api;
    yue2_error_v1 error = { sizeof error };
    float * planar;
    yue2_transcriber_config_v1 config = { sizeof config };
    yue2_transcription_request_v1 request = { sizeof request };
    yue2_transcription_result_v1 result = { sizeof result };
    yue2_transcriber_context * context = NULL;
    int index;
    yue2_status_v1 status;

    if (argc < 2 || argc > 3) {
        fprintf(stderr, "usage: yue2-c-transcription-test TRANSCRIPTION.gguf [DEVICE]\n");
        return 2;
    }
    api = yue2_get_api(YUE2_ABI_VERSION_1);
    if (!api || api->size < YUE2_API_V1_MIN_SIZE) return 1;
    api->error_init(&error);
    planar = (float *)malloc((size_t)frames * channels * sizeof(float));
    if (!planar) return 1;
    for (index = 0; index < frames; ++index) {
        const float value = (float)(sin(2.0 * 3.14159265358979323846 * 220.0 * index /
                                       sample_rate) * 0.25);
        planar[index] = value;
        planar[frames + index] = value * 0.5f;
    }

    api->transcriber_config_init(&config);
    config.model_path = argv[1];
    if (argc == 3) config.device = argv[2];
    status = api->transcriber_create(&config, &context, &error);
    if (status != YUE2_STATUS_OK_V1) {
        fprintf(stderr, "create failed (%d): %s\n", status, error.message);
        free(planar);
        return 1;
    }

    api->transcription_request_init(&request);
    request.audio.samples = planar;
    request.audio.frame_count = frames;
    request.audio.sample_rate = sample_rate;
    request.audio.channels = channels;
    request.audio.layout = YUE2_AUDIO_PLANAR_V1;
    request.melody_only = 0;
    request.window_seconds = 0.2f;
    request.overlap_seconds = 0.0f;
    request.lookahead_seconds = 0.0f;
    request.max_tokens = 32;
    request.on_progress = on_progress;
    request.should_cancel = should_cancel;

    api->transcription_result_init(&result);
    status = api->transcribe(context, &request, &result, &error);
    if (status != YUE2_STATUS_OK_V1 || !result.abc ||
        strncmp(result.abc, "X:1\n", 4) != 0 || !midi_file(result.midi, result.midi_size) ||
        !midi_file(result.melody_midi, result.melody_midi_size) ||
        !midi_file(result.vocal_midi, result.vocal_midi_size) ||
        !midi_file(result.instrumental_midi, result.instrumental_midi_size) ||
        !midi_file(result.chords_midi, result.chords_midi_size) ||
        !result.events_json || strstr(result.events_json, "\"events\"") == NULL ||
        result.duration_seconds < 0.199 || result.duration_seconds > 0.201 ||
        progress_calls == 0) {
        fprintf(stderr, "transcribe failed or returned invalid data (%d): %s\n",
                status, error.message);
        api->transcription_result_free(&result);
        api->transcriber_destroy(context);
        free(planar);
        return 1;
    }
    printf("C ABI V1 transcription events=%llu midi=%llu duration=%.3f\n",
           (unsigned long long)result.event_count,
           (unsigned long long)result.midi_size,
           result.duration_seconds);
    api->transcription_result_free(&result);

    cancel_now = 1;
    status = api->transcribe(context, &request, &result, &error);
    if (status != YUE2_STATUS_CANCELLED_V1 || result.abc != NULL) {
        fprintf(stderr, "cooperative transcription cancellation failed (%d): %s\n",
                status, error.message);
        api->transcription_result_free(&result);
        api->transcriber_destroy(context);
        free(planar);
        return 1;
    }
    api->transcription_result_free(&result);
    api->transcriber_destroy(context);
    free(planar);
    return 0;
}
