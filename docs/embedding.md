# Embedding yue2.cpp

`yue2.dll` / `libyue2.so` exposes a pure-C API for gary4local, JUCE, and other
hosts. New integrations should include
[`include/yue2/c_api_v1.h`](../include/yue2/c_api_v1.h), resolve
`yue2_get_api(YUE2_ABI_VERSION_1)`, and initialize every structure through the
returned table.

The complete versioning, ownership, and callback rules live in
[`C_ABI_V1.md`](C_ABI_V1.md).

## Transcribe audio already owned by the host

Transcription and generation have separate resident contexts. A plugin can load
only SheetSage2/MERT2 for audio-to-score and leave the much larger generator on
a remote service.

```c
const yue2_api_v1 *api = yue2_get_api(YUE2_ABI_VERSION_1);

yue2_error_v1 error = { sizeof error };
api->error_init(&error);

yue2_transcriber_config_v1 config = { sizeof config };
api->transcriber_config_init(&config);
config.model_path = "sheetsage2-mert2-0.7B-v1.0-F16.gguf";
config.device = "cuda";

yue2_transcriber_context *context = NULL;
if (api->transcriber_create(&config, &context, &error) != YUE2_STATUS_OK_V1) {
    log_error(error.message);
    return;
}

yue2_transcription_request_v1 request = { sizeof request };
api->transcription_request_init(&request);
request.audio.samples = planar_samples;
request.audio.frame_count = frames_per_channel;
request.audio.channels = channel_count;
request.audio.sample_rate = host_sample_rate;
request.audio.layout = YUE2_AUDIO_PLANAR_V1;
request.melody_only = 0;

yue2_transcription_result_v1 result = { sizeof result };
api->transcription_result_init(&result);
if (api->transcribe(context, &request, &result, &error) == YUE2_STATUS_OK_V1) {
    use_score(result.abc);
    use_midi(result.midi, result.midi_size);
    use_midi(result.chords_midi, result.chords_midi_size);
}
api->transcription_result_free(&result);
api->transcriber_destroy(context);
```

Input may be planar (the native shape of a JUCE `AudioBuffer`) or interleaved.
The library downmixes without peak normalization and resamples internally.

## Plan, edit, and render

A generator context owns the combined AR/NAR GGUF, VAE GGUF, tokenizer, backend,
and optional resident LoRA stack. Per-call sampling, score controls, callbacks,
and seeds live in `yue2_generation_request_v1`.

```c
yue2_generator_config_v1 config = { sizeof config };
api->generator_config_init(&config);
config.model_path = "yue2-3.6B-v1.0-Q4_K_M.gguf";
config.vae_path = "yue2-vae-v1.0-F16.gguf";
config.tokenizer_path = "qwen.tiktoken";
config.device = "cuda";

yue2_generator_context *generator = NULL;
if (api->generator_create(&config, &generator, &error) != YUE2_STATUS_OK_V1) {
    log_error(error.message);
    return;
}

yue2_generation_request_v1 request = { sizeof request };
api->generation_request_init(&request);
request.style = "dry acoustic trio, close room";
request.instrumental = 1;

yue2_plan_result_v1 plan = { sizeof plan };
api->plan_result_init(&plan);
if (api->plan(generator, &request, &plan, &error) == YUE2_STATUS_OK_V1) {
    offer_midi_drag(plan.midi, plan.midi_size);
    const char *edited_abc = edit_score(plan.abc);
    request.abc = edited_abc;
}

yue2_generation_result_v1 song = { sizeof song };
api->generation_result_init(&song);
if (api->generate(generator, &request, &song, &error) == YUE2_STATUS_OK_V1) {
    play_interleaved(song.samples, song.frame_count,
                     song.channels, song.sample_rate);
    offer_midi_drag(song.midi, song.midi_size);
}
api->generation_result_free(&song);
api->plan_result_free(&plan);
api->generator_destroy(generator);
```

Planning and generation return the same combined, melody, vocal, instrumental,
and chord MIDI family as transcription. Generation MIDI is rebuilt from the
final ABC, so an outro fit or continuation is reflected in the dragged file.

All returned strings, MIDI buffers, token arrays, latents, and PCM are owned by
the shared library. Always use the matching result-free function, including on
cancelled or failed calls.

`transcriber_unload()` and `generator_unload()` release model/VRAM allocations
without discarding the context configuration. The next operation reloads the
same model set lazily. Context calls are not reentrant; serialize them through a
worker queue.

The individual functions in [`c_api.h`](../include/yue2/c_api.h) are retained
for compatibility. They are not the contract new downstream integrations should
target.
