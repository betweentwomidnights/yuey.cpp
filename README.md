# yue2.cpp

Native GGML inference for the YuE2 music ecosystem, built on the same shared
GGML fork used by `sa3.cpp` and `acestep.cpp`.

The first product milestone is audio-to-score transcription:

```text
audio -> 24 kHz mono -> MERT-v2-FullSong -> SheetSage2 -> ABC/MIDI/events
```

The resulting melody-only or full ABC can be passed directly to the native
YuE2 score-conditioned generation pipeline.

## Status

The first native end-to-end path is working. It includes WAV decoding,
downmixing and resampling; the MERT2 log-mel front end, ConvNeXt subsampler and
24 Conformer blocks; SheetSage2's learned layer mixture and encoder projection;
the six-layer BART decoder; grammar-constrained greedy generation; and
ABC/MIDI/event export. Encoder memory and decoder logits have been checked
numerically against the pinned Python reference, and a real-weight API/CLI
smoke test runs with no Python process in the inference path.

The generation-side model boundary is also in place. A native converter emits
the released YuE2-3B checkpoint as BF16 GGUF and the YuE2-VAE as F16 GGUF,
folding all Oobleck weight-normalization pairs and packaging the tokenizer and
JSON sidecars. A metadata-only C++ inspector validates the exact architecture,
key tensor shapes, storage types, checkpoint hashes, and VAE geometry without
allocating model payloads. The native six-stage Oobleck decoder is working on
CPU and CUDA, including bounded 1024-frame tiles with the released 16-frame
halo. The checkpoint-native Qwen tokenizer and prompt protocol now match the
official implementation token-for-token, and the native AR transformer matches
the released model's full 184,704-logit output. Request-local F16 KV caches,
off/ABC/semantic vocabulary masks, windowed repetition penalty, top-k/top-p
sampling, and paired classifier-free guidance are implemented. NAR/flow
execution now runs the released 28-layer flow branch with midpoint ODE solving
and feeds its 64-channel latents into the VAE. `GenerationPipeline` exposes the
complete text/score-to-waveform path while retaining loaded models across
requests, and `yue2-generate` provides the corresponding CLI. PEFT-style YuE2
LoRAs can be converted to a dedicated GGUF and applied functionally across the
AR and NAR projections without modifying or requantizing the base weights.

Decoder generation uses a persistent self-attention KV cache and precomputed
encoder-attention keys and values. Whole-song input uses the released
right-lookahead/overlap plan, re-encodes accepted overlap events as the next
window's decoder prefix, and retains per-window token provenance. Input is
currently WAV-only at the CLI boundary. ABC export follows YuE2's native
two-voice dialect, completes every measure, splits long durations with ties,
mirrors decoded key changes across voices, and places chords in Vocal only.
Full mode reconstructs downbeat-bounded measures and meter changes from the
decoded meter/eighth-position stream, including pickup and final partial-bar
padding. Melody-only mode necessarily falls back to a 4/4, C-major grid because
those fields are not requested from the model.

## Build

```bash
git submodule update --init --recursive
cmake -S . -B build -DYUE2_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=native
cmake --build build --config Release -j
ctest --test-dir build -C Release --output-on-failure
```

Backend options are `YUE2_CUDA`, `YUE2_VULKAN`, `YUE2_METAL`, and `YUE2_HIP`.
They forward to the pinned GGML submodule instead of selecting a second GGML
distribution.

Convert the official transcription checkpoints after downloading their actual
Git-LFS payloads:

```bash
python -m pip install numpy safetensors gguf
python tools/convert_sheetsage2_gguf.py \
  --sheetsage models/SheetSage2 \
  --mert models/MERT-v2-FullSong \
  --out models
```

Given a directory, converters write the GGUF naming convention shared with
`sa3.cpp` and `audiocraft.cpp`, here `sheetsage2-mert2-0.7B-v1.0-F16.gguf`; an
explicit `.gguf` path is used as given. See
[docs/distribution.md](docs/distribution.md).

The converter validates the released architecture, tokenizer fingerprint,
pinned MERT revision and MERT SHA-256, then merges all 96 attention LoRA
projections in float32 before emitting the model.

The converter has also been exercised against the pinned real checkpoints. It
emits 1,039 tensors (96 merged projections) as a roughly 1.26 GiB f16/f32 GGUF.

Package the released generation checkpoints without routing them through a
second framework or GGML distribution:

```bash
python -m pip install numpy torch safetensors gguf
python tools/convert_yue2_gguf.py \
  --model models/YuE2-3B \
  --vae models/YuE2-VAE \
  --out models/YuE2-3B-GGUF

build/bin/yue2-inspect-generation \
  models/YuE2-3B-GGUF/yue2-3.6B-v1.0-BF16.gguf \
  models/YuE2-3B-GGUF/yue2-vae-v1.0-F16.gguf
```

The defaults preserve the main model's released BF16 weights, cast the VAE to
F16 after folding its 88 weight-normalization pairs, copy the Qwen tiktoken and
configuration sidecars, and embed SHA-256 fingerprints of both source
checkpoints. `--model-type` and `--vae-type` also accept `f32`, `f16`, or
`bf16`. Existing outputs require an explicit `--overwrite`. `--model-revision`
and `--vae-revision` record the downloaded Hugging Face revisions in the
standard `general.base_model.*` fields.

## Quantization

`yue2-quantize` re-encodes the generation GGUF with the same mixes as
`sa3-quantize`. It streams one tensor at a time, so host memory stays bounded by
the largest tensor rather than the 7 GB model:

```bash
build/bin/yue2-quantize \
  --in models/YuE2-3B-GGUF/yue2-3.6B-v1.0-BF16.gguf \
  --mix q4_k_m
build/bin/yue2-quant-check \
  --ref models/YuE2-3B-GGUF/yue2-3.6B-v1.0-BF16.gguf \
  --quant models/YuE2-3B-GGUF/yue2-3.6B-v1.0-Q4_K_M.gguf
```

`--mix` accepts `q4_k_m`, `q5_k_m`, `q8_0`, `f16`, and `f32`; without `--out`
the Encoding field of the input name is replaced. The K-quant mixes store
attention values, MLP down projections, the embedding, and the LM head as
Q6_K. Norms, biases, and the flow branch's latent bridges, timestep MLP, and
latent position table keep their source storage. Metadata, including the
checkpoint fingerprint LoRA adapters bind to, is preserved.
`yue2-quant-check` dequantizes every converted tensor and exits nonzero if any
falls below the cosine threshold.

The VAE and the transcription model are refused: both stay F16/F32. The
1.26 GiB F16 transcription model already runs on an 8 GB laptop GPU. Quantized
generation tiers have not yet been validated against real-weight parity and
listening tests.

Convert a PEFT-style generation adapter and optionally bind it to the exact
source-checkpoint fingerprint embedded in the base GGUF:

```bash
python tools/convert_yue2_lora.py \
  --input adapters/my-style/adapter_model.safetensors \
  --config adapters/my-style/adapter_config.json \
  --base-sha256 <sha256-of-model.safetensors> \
  --type f16 \
  --output models/my-style-v1.0-F16-LoRA.gguf
```

The converter accepts YuE2 AR/NAR attention and MLP projections, `lm_head`,
both latent bridges, and the timestep MLP. It rejects unknown targets,
incomplete pairs, mixed ranks, and inconsistent PEFT metadata. At runtime the
adapter stays as a small resident buffer on the base model's backend and is
evaluated as `W*x + (alpha/rank)*strength*B*(A*x)`. This also works with a
quantized base because `W` is never merged or rewritten.

Library clients can construct `yue2::VaeDecoder` from the VAE GGUF and pass a
row-major `[latent_frames,64]` float buffer to `decode()`. The result is
interleaved 48 kHz stereo PCM. Long inputs are tiled automatically to bound
activation memory without crossfades or seam smoothing; `VaeRuntimeOptions`
controls the device, CPU thread count, core/halo sizes, and the diagnostic
full-graph opt-out. Calls on one instance are safe and serialize its mutable
GGML scheduler.

Generation clients can construct `yue2::TextTokenizer` from the packaged
`qwen.tiktoken`, build checkpoint-native positive/negative prefixes with
`make_positive_prefix()` and `make_negative_prefix()`, and generate symbolic or
semantic tokens with `yue2::AutoregressiveModel::generate()` or
`generate_cfg()`. Streaming/server code can retain an
`AutoregressiveSession`, append prompt or sampled tokens to its bounded KV
cache, and read next-token logits without replaying the prefix. Text is
NFC-normalized and always uses ordinary-token encoding: literal strings such
as `<abc>` are not silently promoted to protocol controls. `logits()` remains
available as a cache-free parity/debug primitive. Generated results exclude
the phase end token and report whether it was reached.

For the complete path, construct `yue2::GenerationPipeline` once from the main
GGUF, VAE GGUF, and packaged `qwen.tiktoken`, then call `generate()` with a
`SongRequest`. The result contains the used/generated ABC, raw semantic codec
IDs, time-major 64-channel latents, and interleaved 48 kHz stereo PCM. The
pipeline keeps both models loaded for repeated server or plugin requests.
Seeded flow noise uses the standard C++ generator used by the native runtime;
the explicit-noise `synthesize_latents()` overload is the bit-reproducible
boundary for upstream parity fixtures.

Generate a WAV with an existing full ABC score:

```bash
build/bin/yue2-generate \
  --model models/YuE2-3B-GGUF/yue2-3.6B-v1.0-BF16.gguf \
  --vae models/YuE2-3B-GGUF/yue2-vae-v1.0-F16.gguf \
  --tokenizer models/YuE2-3B-GGUF/qwen.tiktoken \
  --style "acoustic, intimate" \
  --lyrics-file lyrics.txt \
  --symbolic full \
  --abc score.abc \
  --lora models/my-style-v1.0-F16-LoRA.gguf=0.8 \
  --output song.wav \
  --device cuda
```

Omit `--abc` to let the AR branch plan a score, or use `--symbolic off` for
text-only generation. The CLI also exposes seed, guidance, ABC/semantic token
limits, semantic sampling, ODE steps, and CPU thread controls. Repeat `--lora
PATH[=SCALE]` to stack additive adapters.

Run a single-window transcription:

```bash
build/bin/yue2-transcribe \
  --model models/sheetsage2-mert2-0.7B-v1.0-F16.gguf \
  --audio input.wav \
  --output score.abc \
  --midi score.mid \
  --events events.json \
  --device cuda \
  --overlap-seconds 200 \
  --lookahead-seconds 100
```

Both modes retain the vocal and instrumental melody tracks. Use `--full` to
also decode meter, structure, key, and chords; the default melody-only prompt
omits those annotations for YuE2 melody conditioning. Backend selection
uses `YUE2_DEVICE=cpu` to force CPU, `YUE2_DEVICE=cuda` to require CUDA, or
`YUE2_GPU=<index-or-name>` to select a registered accelerator. Explicit
accelerator requests fail if the device cannot be found or initialized; they
never silently fall back to CPU. The JSON output retains raw generated token IDs plus
typed timestamps, meter positions, structure, key, chord, note track, duration,
and interpolated note-end times for integration without reparsing ABC.

The converter's default F16 GGUF is the interactive CUDA choice. Use
`--keep-f32` when building a reference-quality GGUF and `YUE2_DEVICE=cpu` when
exact symbolic agreement with the official FP32 implementation matters more
than latency. On the 60-second oracle fixture, that path produces an exactly
matching ABC score; CUDA/F16 remains numerically close but can choose different
greedy tokens on long material.

Library clients can call `Transcriber::transcribe_mono()` with an existing mono
float PCM buffer and its sample rate. This bypasses file I/O and is the intended
gary4juce/server boundary; the runtime performs the same amplitude-preserving
24 kHz resampling and returns the same tokens, events, ABC, and MIDI. Pass
`TranscriberRuntimeOptions` at construction to select a device and CPU thread
count per model instance without changing process-global environment state.
Concurrent calls on one instance are safe and serialize its mutable GGML
scheduler; use separate instances only when the host intentionally wants
parallel model execution and can afford the duplicated weights.

Hosts that need a stable module boundary can instead link `yue2.dll` or
`libyue2.so` through the pure-C API in `include/yue2/c_api.h`. It exposes
separate resident transcription and generation contexts, accepts interleaved or
JUCE-style planar input PCM, and returns library-owned ABC/MIDI/events or
waveform/intermediate buffers with matching free functions. Progress and
cooperative cancellation are available at AR-token, flow-step, VAE-tile, and
transcription-window boundaries. See [docs/embedding.md](docs/embedding.md).

## Models and licenses

This repository contains engine code only. It does not redistribute model
weights. The released YuE2, SheetSage2, and MERT2 checkpoints are licensed
under CC BY-NC 4.0; users must obtain them from their upstream publishers and
comply with their terms. See [docs/reference-lock.md](docs/reference-lock.md).

Engine code in this repository is MIT licensed. Upstream code consulted as a
reference retains its own license.
