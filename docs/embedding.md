# Embedding yue2.cpp

`yue2.dll` / `libyue2.so` exposes a pure-C ABI for gary4local, JUCE, and
other hosts that cannot safely exchange C++ standard-library objects across a
module boundary. The public contract is
[`include/yue2/c_api.h`](../include/yue2/c_api.h).

Transcription and generation use separate contexts. This is deliberate: a host
that only needs audio-to-score loads the SheetSage2/MERT2 GGUF without also
paying for the 3B generation model and VAE. A context owns its model weights and
backend scheduler, remains resident across calls, and is not reentrant. Use one
worker queue per context, or create multiple contexts only when the host can
afford duplicate weights.

## In-memory transcription

The transcription request accepts float PCM as either ordinary interleaved
frames or channel-planar buffers (the native shape of a JUCE `AudioBuffer`). It
downmixes without normalization, resamples internally to 24 kHz, and returns
ABC, Standard MIDI bytes, and a lossless events JSON document.

```c
#include "yue2/c_api.h"

char error[512] = {0};
yue2_transcriber_config config = {0};
config.size = sizeof config;
config.model_path = "sheetsage2-mert2-f16.gguf";
config.device = "cuda";
yue2_transcriber_context *ctx =
    yue2_transcriber_create(&config, error, sizeof error);

yue2_transcription_request request = {0};
request.size = sizeof request;
request.samples = planar_samples;
request.frame_count = frames_per_channel;
request.channels = channel_count;
request.sample_rate = host_sample_rate;
request.layout = YUE2_AUDIO_PLANAR;

yue2_transcription_result result = {0};
result.size = sizeof result;
if (yue2_transcribe(ctx, &request, &result, error, sizeof error) == 0) {
    use_abc(result.abc);
    use_midi(result.midi, result.midi_size);
    yue2_free_transcription_result(&result);
}
yue2_transcriber_free(ctx);
```

## Resident generation

The generator context loads the combined AR/NAR GGUF, VAE GGUF, and tokenizer
once. Each request may use an external ABC score, ask the AR model to plan one,
or disable symbolic conditioning. Token budgets, sampling, guidance, and flow
steps can vary per call without reloading the model. Results contain interleaved
48 kHz stereo PCM plus ABC, raw semantic codec IDs, and `[frames,64]` latents.

Generation LoRAs are fixed when the resident context is created. They remain
unmerged on the same backend as the base model, so hosts can use F16 or F32
adapter factors with an unmodified quantized base:

```c
yue2_lora_adapter adapters[2] = {0};
adapters[0].size = sizeof adapters[0];
adapters[0].path = "genre.gguf";
adapters[0].strength = 0.8f;
adapters[1].size = sizeof adapters[1];
adapters[1].path = "instrument.gguf";
adapters[1].strength = 0.5f;

yue2_generator_config config = {0};
config.size = sizeof config;
config.model_path = "yue2-3b-bf16.gguf";
config.vae_path = "yue2-vae-f16.gguf";
config.tokenizer_path = "qwen.tiktoken";
config.device = "cuda";
config.lora_adapters = adapters;
config.lora_adapter_count = 2;
```

The adapter converter and CLI syntax are documented in the main README. A
checkpoint fingerprint in an adapter must match the base GGUF; a zero strength
still validates and loads the adapter but bypasses its graph operations.

All result pointers are allocated by the shared library. Release them only with
`yue2_free_transcription_result()` or `yue2_free_generation_result()`; this
keeps the allocator/CRT boundary correct on Windows.

Both requests accept synchronous progress and cooperative-cancel callbacks.
Generation reports `abc`, `semantic`, `flow`, `decode`, and `complete` stages;
transcription reports completed windows. A callback runs on the inference
thread, so it should only update atomics or enqueue a small message. Cancellation
is checked between AR tokens, flow steps, VAE tiles, and transcription windows;
an already-running backend graph completes before the call returns.

The CMake target is enabled by `YUE2_BUILD_SHARED=ON` (the default). Windows
produces `yue2.dll` plus `yue2-dll.lib`, avoiding a collision with the static
`yue2.lib`; Unix produces `libyue2.so` alongside `libyue2.a`.
