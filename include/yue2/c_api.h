/* Stable C ABI for embedding yue2.cpp in gary4local, JUCE, and other hosts.
 *
 * Contexts are intentionally split: transcription and generation have
 * independent model sets and can be loaded or released separately. Calls on
 * one context are not reentrant. No C++ exception crosses this boundary.
 * Every allocated result must be released by its matching yue2_free_* call so
 * allocation and deallocation happen inside the same runtime/DLL.
 */
#ifndef YUE2_C_API_H
#define YUE2_C_API_H

#include <stdint.h>

#if defined(_WIN32) && defined(YUE2_BUILD_DLL)
#  define YUE2_API __declspec(dllexport)
#else
#  define YUE2_API
#endif

#ifdef __cplusplus
extern "C" {
#endif

typedef struct yue2_transcriber_context yue2_transcriber_context;
typedef struct yue2_generator_context yue2_generator_context;

/* Called synchronously on the inference thread. fraction is stage-local 0..1.
 * Generation stages are "abc", "semantic", "flow", "decode", "complete";
 * transcription uses "transcription". */
typedef void (*yue2_progress_cb)(
    void * user,
    const char * stage,
    uint32_t current,
    uint32_t total,
    float fraction);
typedef int32_t (*yue2_cancel_cb)(void * user);

enum {
    YUE2_AUDIO_INTERLEAVED = 0,
    YUE2_AUDIO_PLANAR = 1,
};

enum {
    YUE2_SYMBOLIC_FULL = 0,
    YUE2_SYMBOLIC_MELODY = 1,
    YUE2_SYMBOLIC_OFF = 2,
};

enum {
    YUE2_TRANSCRIPTION_STANDARD = 0,
    YUE2_TRANSCRIPTION_PAPER = 1,
};

typedef struct {
    uint32_t size;             /* set to sizeof(yue2_transcriber_config) */
    const char * model_path;   /* required SheetSage2/MERT2 GGUF */
    const char * device;       /* NULL/empty = automatic; "cpu"/"cuda" = required */
    int32_t threads;           /* 0 = backend default */
} yue2_transcriber_config;

typedef struct {
    uint32_t size;             /* set to sizeof(yue2_transcription_request) */
    const float * samples;     /* required float PCM */
    uint64_t frame_count;      /* frames per channel */
    int32_t sample_rate;
    int32_t channels;
    int32_t layout;            /* YUE2_AUDIO_INTERLEAVED or YUE2_AUDIO_PLANAR */

    /* Leave options_set=0 for the native defaults. */
    int32_t options_set;
    int32_t melody_only;
    int32_t preset;            /* YUE2_TRANSCRIPTION_* */
    int32_t timing_set;        /* apply the three timing fields below */
    float window_seconds;
    float overlap_seconds;
    float lookahead_seconds;
    uint64_t max_tokens;
    yue2_progress_cb on_progress;
    yue2_cancel_cb should_cancel;
    void * user;
} yue2_transcription_request;

typedef struct {
    uint32_t size;             /* set to sizeof(yue2_transcription_result) */
    char * abc;
    uint8_t * midi;
    uint64_t midi_size;
    char * events_json;        /* lossless typed events + per-window tokens */
    uint64_t event_count;
    double duration_seconds;
} yue2_transcription_result;

typedef struct {
    uint32_t size;             /* set to sizeof(yue2_lora_adapter) */
    const char * path;         /* required YuE2 LoRA GGUF */
    float strength;            /* finite; zero is a validated no-op */
} yue2_lora_adapter;

typedef struct {
    uint32_t size;             /* set to sizeof(yue2_generator_config) */
    const char * model_path;   /* required combined YuE2 AR/NAR GGUF */
    const char * vae_path;     /* required YuE2 VAE GGUF */
    const char * tokenizer_path; /* required qwen.tiktoken */
    const char * device;
    int32_t threads;
    const yue2_lora_adapter * lora_adapters; /* optional resident adapter stack */
    uint32_t lora_adapter_count;
} yue2_generator_config;

typedef struct {
    uint32_t size;             /* set to sizeof(yue2_generation_request) */
    const char * style;        /* NULL is treated as empty */
    const char * lyrics;       /* NULL is treated as empty */
    const char * abc;          /* optional external score */
    int32_t symbolic_mode;     /* YUE2_SYMBOLIC_*; zero defaults to full */
    uint64_t seed;             /* zero is a valid seed */
    int32_t seed_set;          /* zero -> native default seed 831001 */
    float guidance_scale;
    int32_t guidance_set;

    /* Zero keeps the corresponding native default, except temperature which
     * uses temperature_set so callers can request greedy temperature 0. */
    uint32_t abc_min_tokens;
    uint32_t abc_max_tokens;
    uint32_t semantic_min_tokens;
    uint32_t semantic_max_tokens;
    uint32_t ode_steps;
    float temperature;
    int32_t temperature_set;
    uint32_t top_k;
    float top_p;
    float repetition_penalty;
    uint32_t penalty_window;
    yue2_progress_cb on_progress;
    yue2_cancel_cb should_cancel;
    void * user;
} yue2_generation_request;

typedef struct {
    uint32_t size;             /* set to sizeof(yue2_generation_result) */
    float * interleaved_samples;
    uint64_t frame_count;
    int32_t channels;
    int32_t sample_rate;
    char * abc;
    int32_t * semantic_codec_ids;
    uint64_t semantic_count;
    float * latents;           /* row-major [semantic_count,64] */
    uint64_t latent_count;
    uint64_t seed;
    int32_t abc_truncated;
    int32_t semantic_truncated;
} yue2_generation_result;

YUE2_API yue2_transcriber_context * yue2_transcriber_create(
    const yue2_transcriber_config * config, char * error, int32_t error_size);
YUE2_API int32_t yue2_transcribe(
    yue2_transcriber_context * context,
    const yue2_transcription_request * request,
    yue2_transcription_result * result,
    char * error,
    int32_t error_size);
YUE2_API void yue2_free_transcription_result(yue2_transcription_result * result);
YUE2_API void yue2_transcriber_free(yue2_transcriber_context * context);

YUE2_API yue2_generator_context * yue2_generator_create(
    const yue2_generator_config * config, char * error, int32_t error_size);
YUE2_API int32_t yue2_generate(
    yue2_generator_context * context,
    const yue2_generation_request * request,
    yue2_generation_result * result,
    char * error,
    int32_t error_size);
YUE2_API void yue2_free_generation_result(yue2_generation_result * result);
YUE2_API void yue2_generator_free(yue2_generator_context * context);

YUE2_API const char * yue2_c_version(void);

#ifdef __cplusplus
}
#endif

#endif
