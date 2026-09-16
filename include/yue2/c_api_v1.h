/* yue2.cpp C ABI V1.
 *
 * Resolve yue2_get_api(), initialize public structs through the returned
 * table, and release library-owned results through their matching free
 * function. All structs are size tagged. Top-level structs, callback
 * payloads, and strided array entries may grow by appending fields; types
 * embedded by value are frozen for ABI V1.
 */
#ifndef YUE2_C_API_V1_H
#define YUE2_C_API_V1_H

#include <stddef.h>
#include <stdint.h>

#ifndef YUE2_API
#  if defined(_WIN32) && defined(YUE2_BUILD_DLL)
#    define YUE2_API __declspec(dllexport)
#  else
#    define YUE2_API
#  endif
#endif

#if defined(_WIN32)
#  define YUE2_CALL __cdecl
#else
#  define YUE2_CALL
#endif

#ifdef __cplusplus
extern "C" {
#endif

#ifndef YUE2_CONTEXTS_DECLARED
#define YUE2_CONTEXTS_DECLARED
typedef struct yue2_transcriber_context yue2_transcriber_context;
typedef struct yue2_generator_context yue2_generator_context;
#endif

#define YUE2_ABI_VERSION_1 1u

typedef int32_t yue2_status_v1;
enum {
    YUE2_STATUS_OK_V1               = 0,
    YUE2_STATUS_INVALID_ARGUMENT_V1 = 1,
    YUE2_STATUS_UNSUPPORTED_ABI_V1  = 2,
    YUE2_STATUS_CANCELLED_V1        = 3,
    YUE2_STATUS_MODEL_ERROR_V1      = 4,
    YUE2_STATUS_IO_ERROR_V1         = 5,
    YUE2_STATUS_OUT_OF_MEMORY_V1    = 6,
    YUE2_STATUS_INTERNAL_ERROR_V1   = 7
};

typedef int32_t yue2_audio_layout_v1;
enum {
    YUE2_AUDIO_PLANAR_V1 = 0,
    YUE2_AUDIO_INTERLEAVED_V1 = 1
};

typedef int32_t yue2_symbolic_mode_v1;
enum {
    YUE2_SYMBOLIC_FULL_V1 = 0,
    YUE2_SYMBOLIC_MELODY_V1 = 1,
    YUE2_SYMBOLIC_OFF_V1 = 2
};

typedef int32_t yue2_ending_mode_v1;
enum {
    YUE2_ENDING_NATURAL_V1 = 0,
    YUE2_ENDING_OUTRO_V1 = 1
};

typedef int32_t yue2_transcription_preset_v1;
enum {
    YUE2_TRANSCRIPTION_STANDARD_V1 = 0,
    YUE2_TRANSCRIPTION_PAPER_V1 = 1
};

typedef int32_t yue2_progress_stage_v1;
enum {
    YUE2_PROGRESS_LOADING_V1 = 0,
    YUE2_PROGRESS_TRANSCRIPTION_V1 = 1,
    YUE2_PROGRESS_ABC_V1 = 2,
    YUE2_PROGRESS_SEMANTIC_V1 = 3,
    YUE2_PROGRESS_FLOW_V1 = 4,
    YUE2_PROGRESS_DECODE_V1 = 5,
    YUE2_PROGRESS_DONE_V1 = 6,
    YUE2_PROGRESS_OTHER_V1 = 7
};

typedef struct {
    uint32_t size;
    yue2_status_v1 code;
    char message[1024];
} yue2_error_v1;

typedef struct {
    uint32_t size;
    yue2_progress_stage_v1 stage;
    const char * stage_name; /* borrowed; valid only during the callback */
    uint32_t current;
    uint32_t total;
    float fraction;
} yue2_progress_v1;

typedef void (YUE2_CALL *yue2_progress_callback_v1)(
    void * user, const yue2_progress_v1 * progress);
typedef int32_t (YUE2_CALL *yue2_cancel_callback_v1)(void * user);
typedef void (YUE2_CALL *yue2_reserved_function_v1)(void);

/* Non-owning float PCM. frame_count is per channel. */
typedef struct {
    uint32_t size;
    const float * samples;
    uint64_t frame_count;
    uint32_t channels;
    uint32_t sample_rate;
    yue2_audio_layout_v1 layout;
    uint32_t reserved;
} yue2_audio_view_v1;

typedef struct {
    uint32_t size;
    const char * model_path;
    const char * device;
    int32_t threads;
} yue2_transcriber_config_v1;

typedef struct {
    uint32_t size;
    yue2_audio_view_v1 audio;
    int32_t melody_only;
    yue2_transcription_preset_v1 preset;
    float window_seconds;
    float overlap_seconds;
    float lookahead_seconds;
    uint64_t max_tokens;
    yue2_progress_callback_v1 on_progress;
    yue2_cancel_callback_v1 should_cancel;
    void * callback_user;
} yue2_transcription_request_v1;

/* Each MIDI buffer is a complete format-1 Standard MIDI File. */
typedef struct {
    uint32_t size;
    char * abc;
    uint8_t * midi;
    uint64_t midi_size;
    uint8_t * melody_midi;
    uint64_t melody_midi_size;
    uint8_t * vocal_midi;
    uint64_t vocal_midi_size;
    uint8_t * instrumental_midi;
    uint64_t instrumental_midi_size;
    uint8_t * chords_midi;
    uint64_t chords_midi_size;
    char * events_json;
    uint64_t event_count;
    double duration_seconds;
} yue2_transcription_result_v1;

/* generator_config.adapter_stride must be at least YUE2_ADAPTER_V1_MIN_SIZE. */
typedef struct {
    uint32_t size;
    const char * path;
    float strength;
} yue2_adapter_v1;

typedef struct {
    uint32_t size;
    const char * model_path;
    const char * vae_path;
    const char * tokenizer_path;
    const char * device;
    int32_t threads;
    const yue2_adapter_v1 * adapters;
    uint32_t adapter_count;
    uint32_t adapter_stride;
} yue2_generator_config_v1;

/* Initialized values are the native generation defaults. abc and abc_prefix
 * are mutually exclusive. An external abc skips symbolic planning. */
typedef struct {
    uint32_t size;
    const char * style;
    const char * lyrics;
    const char * abc;
    const char * abc_prefix;
    yue2_symbolic_mode_v1 symbolic_mode;
    uint64_t seed;
    float guidance_scale;
    int32_t guidance_set;

    uint32_t abc_min_tokens;
    uint32_t abc_max_tokens;
    float abc_temperature;
    uint32_t abc_top_k;
    float abc_top_p;
    float abc_repetition_penalty;
    uint32_t abc_penalty_window;

    uint32_t semantic_min_tokens;
    uint32_t semantic_max_tokens;
    int32_t semantic_budget_explicit;
    float semantic_temperature;
    uint32_t semantic_top_k;
    float semantic_top_p;
    float semantic_repetition_penalty;
    uint32_t semantic_penalty_window;
    uint32_t ode_steps;

    int32_t experimental_vocal_rest;
    int32_t instrumental;
    uint32_t target_bars;
    yue2_ending_mode_v1 ending_mode;
    uint32_t outro_bars;

    yue2_progress_callback_v1 on_progress;
    yue2_cancel_callback_v1 should_cancel;
    void * callback_user;
} yue2_generation_request_v1;

typedef struct {
    uint32_t size;
    char * abc;
    int32_t * abc_token_ids;
    uint64_t abc_token_count;
    uint64_t seed;
    uint32_t score_bars;
    double score_duration_seconds;
    int32_t abc_truncated;
} yue2_plan_result_v1;

/* Library-owned interleaved float audio plus the score and intermediate
 * representations required by editor and remote-service clients. */
typedef struct {
    uint32_t size;
    float * samples;
    uint64_t frame_count;
    uint32_t channels;
    uint32_t sample_rate;
    char * abc;
    int32_t * abc_token_ids;
    uint64_t abc_token_count;
    int32_t * semantic_codec_ids;
    uint64_t semantic_count;
    float * latents; /* row-major [semantic_count, 64] */
    uint64_t latent_count;
    uint64_t seed;
    uint32_t score_bars;
    double score_duration_seconds;
    uint32_t semantic_budget;
    int32_t abc_truncated;
    int32_t semantic_truncated;
} yue2_generation_result_v1;

typedef struct yue2_api_v1 {
    uint32_t size;
    uint32_t abi_version;
    const char * (YUE2_CALL *runtime_version)(void);

    void (YUE2_CALL *error_init)(yue2_error_v1 * error);
    void (YUE2_CALL *audio_view_init)(yue2_audio_view_v1 * audio);
    void (YUE2_CALL *transcriber_config_init)(yue2_transcriber_config_v1 * config);
    void (YUE2_CALL *transcription_request_init)(yue2_transcription_request_v1 * request);
    void (YUE2_CALL *transcription_result_init)(yue2_transcription_result_v1 * result);
    void (YUE2_CALL *adapter_init)(yue2_adapter_v1 * adapter);
    void (YUE2_CALL *generator_config_init)(yue2_generator_config_v1 * config);
    void (YUE2_CALL *generation_request_init)(yue2_generation_request_v1 * request);
    void (YUE2_CALL *plan_result_init)(yue2_plan_result_v1 * result);
    void (YUE2_CALL *generation_result_init)(yue2_generation_result_v1 * result);

    yue2_status_v1 (YUE2_CALL *transcriber_create)(
        const yue2_transcriber_config_v1 * config,
        yue2_transcriber_context ** out_context,
        yue2_error_v1 * error);
    void (YUE2_CALL *transcriber_unload)(yue2_transcriber_context * context);
    void (YUE2_CALL *transcriber_destroy)(yue2_transcriber_context * context);
    yue2_status_v1 (YUE2_CALL *transcribe)(
        yue2_transcriber_context * context,
        const yue2_transcription_request_v1 * request,
        yue2_transcription_result_v1 * result,
        yue2_error_v1 * error);
    void (YUE2_CALL *transcription_result_free)(yue2_transcription_result_v1 * result);

    yue2_status_v1 (YUE2_CALL *generator_create)(
        const yue2_generator_config_v1 * config,
        yue2_generator_context ** out_context,
        yue2_error_v1 * error);
    void (YUE2_CALL *generator_unload)(yue2_generator_context * context);
    void (YUE2_CALL *generator_destroy)(yue2_generator_context * context);
    yue2_status_v1 (YUE2_CALL *plan)(
        yue2_generator_context * context,
        const yue2_generation_request_v1 * request,
        yue2_plan_result_v1 * result,
        yue2_error_v1 * error);
    void (YUE2_CALL *plan_result_free)(yue2_plan_result_v1 * result);
    yue2_status_v1 (YUE2_CALL *generate)(
        yue2_generator_context * context,
        const yue2_generation_request_v1 * request,
        yue2_generation_result_v1 * result,
        yue2_error_v1 * error);
    void (YUE2_CALL *generation_result_free)(yue2_generation_result_v1 * result);

    yue2_reserved_function_v1 reserved[16];
} yue2_api_v1;

#define YUE2_ERROR_V1_MIN_SIZE \
    ((uint32_t)(offsetof(yue2_error_v1, message) + sizeof(((yue2_error_v1*)0)->message)))
#define YUE2_PROGRESS_V1_MIN_SIZE \
    ((uint32_t)(offsetof(yue2_progress_v1, fraction) + sizeof(((yue2_progress_v1*)0)->fraction)))
#define YUE2_AUDIO_VIEW_V1_MIN_SIZE \
    ((uint32_t)(offsetof(yue2_audio_view_v1, reserved) + sizeof(((yue2_audio_view_v1*)0)->reserved)))
#define YUE2_TRANSCRIBER_CONFIG_V1_MIN_SIZE \
    ((uint32_t)(offsetof(yue2_transcriber_config_v1, threads) + sizeof(((yue2_transcriber_config_v1*)0)->threads)))
#define YUE2_TRANSCRIPTION_REQUEST_V1_MIN_SIZE \
    ((uint32_t)(offsetof(yue2_transcription_request_v1, callback_user) + sizeof(((yue2_transcription_request_v1*)0)->callback_user)))
#define YUE2_TRANSCRIPTION_RESULT_V1_MIN_SIZE \
    ((uint32_t)(offsetof(yue2_transcription_result_v1, duration_seconds) + sizeof(((yue2_transcription_result_v1*)0)->duration_seconds)))
#define YUE2_ADAPTER_V1_MIN_SIZE \
    ((uint32_t)(offsetof(yue2_adapter_v1, strength) + sizeof(((yue2_adapter_v1*)0)->strength)))
#define YUE2_GENERATOR_CONFIG_V1_MIN_SIZE \
    ((uint32_t)(offsetof(yue2_generator_config_v1, adapter_stride) + sizeof(((yue2_generator_config_v1*)0)->adapter_stride)))
#define YUE2_GENERATION_REQUEST_V1_MIN_SIZE \
    ((uint32_t)(offsetof(yue2_generation_request_v1, callback_user) + sizeof(((yue2_generation_request_v1*)0)->callback_user)))
#define YUE2_PLAN_RESULT_V1_MIN_SIZE \
    ((uint32_t)(offsetof(yue2_plan_result_v1, abc_truncated) + sizeof(((yue2_plan_result_v1*)0)->abc_truncated)))
#define YUE2_GENERATION_RESULT_V1_MIN_SIZE \
    ((uint32_t)(offsetof(yue2_generation_result_v1, semantic_truncated) + sizeof(((yue2_generation_result_v1*)0)->semantic_truncated)))
#define YUE2_API_V1_MIN_SIZE \
    ((uint32_t)(offsetof(yue2_api_v1, generation_result_free) + sizeof(((yue2_api_v1*)0)->generation_result_free)))

/* The sole symbol a dynamically loaded V1 host needs to resolve. The table is
 * static and remains owned by the library until the module is unloaded. */
YUE2_API const yue2_api_v1 * YUE2_CALL yue2_get_api(uint32_t abi_version);

#ifdef __cplusplus
}
#endif

#endif /* YUE2_C_API_V1_H */
