#include "yue2/c_api.h"

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int progress_calls = 0;
static int cancel_now = 0;

static void on_progress(
    void * user, const char * stage, uint32_t current, uint32_t total, float fraction) {
    (void)user;
    if (strcmp(stage, "transcription") == 0 && current <= total &&
        fraction >= 0.0f && fraction <= 1.0f) {
        ++progress_calls;
    }
}

static int32_t should_cancel(void * user) {
    (void)user;
    return cancel_now;
}

int main(int argc, char ** argv) {
    enum { frames = 4800, channels = 2, sample_rate = 24000 };
    char error[512] = {0};
    float * planar;
    yue2_transcriber_config config;
    yue2_transcription_request request;
    yue2_transcription_result result;
    yue2_transcriber_context * context;
    int index;
    int rc;

    if (argc < 2 || argc > 3) {
        fprintf(stderr, "usage: yue2-c-transcription-test TRANSCRIPTION.gguf [DEVICE]\n");
        return 2;
    }
    planar = (float *)malloc((size_t)frames * channels * sizeof(float));
    if (!planar) return 1;
    for (index = 0; index < frames; ++index) {
        const float value = (float)(sin(2.0 * 3.14159265358979323846 * 220.0 * index /
                                       sample_rate) * 0.25);
        planar[index] = value;
        planar[frames + index] = value * 0.5f;
    }

    memset(&config, 0, sizeof config);
    config.size = (uint32_t)sizeof config;
    config.model_path = argv[1];
    if (argc == 3) config.device = argv[2];
    context = yue2_transcriber_create(&config, error, (int32_t)sizeof error);
    if (!context) {
        fprintf(stderr, "create failed: %s\n", error);
        free(planar);
        return 1;
    }

    memset(&request, 0, sizeof request);
    request.size = (uint32_t)sizeof request;
    request.samples = planar;
    request.frame_count = frames;
    request.sample_rate = sample_rate;
    request.channels = channels;
    request.layout = YUE2_AUDIO_PLANAR;
    request.options_set = 1;
    request.melody_only = 1;
    request.preset = YUE2_TRANSCRIPTION_STANDARD;
    request.timing_set = 1;
    request.window_seconds = 0.2f;
    request.overlap_seconds = 0.0f;
    request.lookahead_seconds = 0.0f;
    request.max_tokens = 6;
    request.on_progress = on_progress;
    request.should_cancel = should_cancel;

    memset(&result, 0, sizeof result);
    result.size = (uint32_t)sizeof result;
    rc = yue2_transcribe(context, &request, &result, error, (int32_t)sizeof error);
    if (rc != 0 || !result.abc || strncmp(result.abc, "X:1\n", 4) != 0 ||
        !result.midi || result.midi_size < 22 ||
        memcmp(result.midi, "MThd", 4) != 0 ||
        !result.events_json || strstr(result.events_json, "\"events\"") == NULL ||
        result.duration_seconds < 0.199 || result.duration_seconds > 0.201 ||
        progress_calls == 0) {
        fprintf(stderr, "transcribe failed or returned invalid data (%d): %s\n", rc, error);
        yue2_free_transcription_result(&result);
        yue2_transcriber_free(context);
        free(planar);
        return 1;
    }
    printf("C ABI transcription events=%llu midi=%llu duration=%.3f\n",
           (unsigned long long)result.event_count,
           (unsigned long long)result.midi_size,
           result.duration_seconds);
    yue2_free_transcription_result(&result);

    cancel_now = 1;
    result.size = (uint32_t)sizeof result;
    rc = yue2_transcribe(context, &request, &result, error, (int32_t)sizeof error);
    if (rc == 0 || strstr(error, "cancelled") == NULL || result.abc != NULL) {
        fprintf(stderr, "cooperative transcription cancellation failed (%d): %s\n", rc, error);
        yue2_free_transcription_result(&result);
        yue2_transcriber_free(context);
        free(planar);
        return 1;
    }
    yue2_free_transcription_result(&result);
    yue2_transcriber_free(context);
    free(planar);
    return 0;
}
