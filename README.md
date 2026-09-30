# yuey.cpp

YuE2 in C++ for the gary eco-system

This implementation of YuE2 shares a GGML submodule with
[`betweentwomidnights/acestep.cpp`](https://github.com/betweentwomidnights/acestep.cpp),
[`betweentwomidnights/sa3.cpp`](https://github.com/betweentwomidnights/sa3.cpp),
and
[`betweentwomidnights/audiocraft.cpp`](https://github.com/betweentwomidnights/audiocraft.cpp).

It is mainly designed for downstream usage in
[`betweentwomidnights/gary4juce`](https://github.com/betweentwomidnights/gary4juce),
while remaining usable as a standalone C++ library, CLI, and local web app.

The repository is named **Yuey**, our nickname for the project. Binaries and
public APIs retain the `yue2` prefix to identify the underlying model.

## What works

- Audio transcription to full or melody-only ABC, multi-track MIDI, and timed
  events using native MERT2 + SheetSage2 inference.
- Text-to-music, score-conditioned generation, covers, and real-audio semantic
  continuation through the native YuE2 AR, flow, and VAE pipeline.
- Typed BPM, key, and meter controls that are applied to the planning score
  instead of being left to prompt interpretation.
- Score-first length and ending controls that fit generated plans to musical
  bars and let semantic generation reach the score's natural end.
- Best-effort instrumental generation using score-aligned empty lyric sections,
  a duration-preserving rested Vocal lane, and optional instrumental AR adapter
  support.
- BF16, Q8_0, Q5_K_M, and Q4_K_M generation models, including automatic device
  memory recommendations.
- A local asynchronous HTTP server and embedded Yuey web UI.
- Native C++ and stable C APIs for future DAW and plugin integration.

The complete path is validated on an RTX 5070 Laptop GPU and a DGX Spark using
CUDA. Other backends remain works in progress.

## Quickstart

```bash
git clone --recurse-submodules https://github.com/betweentwomidnights/yuey.cpp.git
cd yuey.cpp
```

### Windows + CUDA

With Visual Studio 2022 C++ tools and the CUDA toolkit installed:

```powershell
.\build.cmd cuda
. .\env.ps1
```

`build.cmd` finds CMake (including Visual Studio's bundled copy), configures,
builds, runs the tests, and writes the environment helper. Use `cpu` or
`vulkan` instead of `cuda` for another Windows backend.

Download the complete Q4_K_M laptop set:

```powershell
.\models.cmd --profile full
```

Use `--encoding q8_0` for the high-quality quantized tier or `--encoding bf16`
for the reference-precision model. Q4_K_M remains the default for 8 GB laptop
GPUs.

Use `--profile core` for generation only, or `transcribe` for generation plus
SheetSage2 transcription. Linux and macOS can run `./models.sh`; the faster SDK
path is `python tools/download_models.py --profile full` after installing
`huggingface_hub[hf_xet]`.

The resulting layout is:

```text
models/
  yue2-3.6B-v1.0-Q4_K_M.gguf
  yue2-vae-v1.0-F16.gguf
  yue2-qwen.tiktoken
  sheetsage2-mert2-0.7B-v1.0-F16.gguf
  yue2-instrumental-cot-full-v1.0-F16-LoRA.gguf
  yue2-realaudio-nar-v9-v1.0-F16-LoRA.gguf
  yue2-semantic-tokenizer-0.7B-v1.0-F16.gguf
```

Start the local app:

```powershell
yue2-server
```

Open <http://127.0.0.1:8007/>. The UI provides generation, transcription,
remixing, piano-roll score editing, MIDI export, and quantization-tier selection.

Run a 16-bar generation directly from the CLI:

```powershell
yue2-generate --encoding q4_k_m --instrumental --prompt "dreamy synth pop" --bars 16 --ending outro --bpm 95 --key "C# minor" --out song.wav
```

Generate only the inexpensive editable plan before committing to audio:

```powershell
yue2-generate --encoding q4_k_m --prompt "dreamy synth pop" --bpm 95 --key "C# minor" --plan-only --score-output song.abc
```

Omit `--instrumental` and use either `--lyrics "..."` or
`--lyrics-file .\lyrics.txt` for a vocal generation.
Instrumental mode can still produce occasional sung material or vocal samples;
review its output before downstream use.

### Linux + CUDA

```bash
cmake -S . -B build-cuda -DYUE2_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=native \
  -DCMAKE_BUILD_TYPE=Release
cmake --build build-cuda --parallel
ctest --test-dir build-cuda --output-on-failure
```

### Apple silicon + Metal

The build needs the `ggml` submodule. If you cloned without
`--recurse-submodules`, run `git submodule update --init --recursive` first.

```bash
cmake -S . -B build-metal -DYUE2_METAL=ON -DCMAKE_BUILD_TYPE=Release
cmake --build build-metal --parallel
ctest --test-dir build-metal --output-on-failure
```

Download the models on the Mac (`./models.sh --profile full` is 5.7 GB for
Q4_K_M) or copy an existing `models/` directory, then start the server from the
repository root:

```bash
build-metal/bin/yue2-server --models-dir models --encoding Q4_K_M
```

Backend options are `YUE2_CUDA`, `YUE2_VULKAN`, `YUE2_METAL`, and `YUE2_HIP`.
Things worth knowing on a Mac:

- ggml turns Metal on by default on macOS, so leaving `YUE2_METAL` off does not
  give a CPU-only build. For a CPU baseline, configure with `-DGGML_METAL=OFF`,
  or keep the Metal build and pass `--device cpu`.
- `--device` (and `YUE2_DEVICE`) takes `cpu`, a device index, or any part of a
  device's name or description. The Metal GPU is named `MTL0`, so use
  `--device mtl`; `--device metal` matches nothing and fails.
  `yue2-server --props` lists the devices.
- Metal reports roughly two thirds of unified memory as its working set (about
  21 GiB on a 32 GB M4), and that is the figure the UI and the tier
  recommendation use.
- Generation is far slower than on CUDA, and the cost grows faster than the
  length of the song. On an M4, transcribing a 3:46 song took 26 s on Metal
  against 96 s on CPU. A transcribe + remix (Q4_K_M) took 109 s for a 30 s clip,
  277 s for 60 s, and 916 s for 120 s, and about 37 minutes for the full song.
  Nearly all of the growth is the flow stage, at 2.3, 6.2, and 22.6 s per step
  for the three clips. Memory was not the limit: swap did not grow and the server
  peaked near 4 GB. Try short clips first.

## TODO

- [ ] Publish the remaining Q5_K_M tier after release listening tests.
- [ ] Fully validate the Vulkan and Metal backends.
- [ ] Produce useful benchmarks on hardware beyond our RTX 5070 Laptop GPU and
  DGX Spark.
- [ ] Validate the C ABI from standalone Ableton Live and REAPER extensions,
  including host-selected MIDI-region workflows.

## Documentation

- [Architecture](docs/architecture.md)
- [Server and web API](docs/server.md)
- [C/C++ embedding](docs/embedding.md)
- [C ABI V1 contract](docs/C_ABI_V1.md)
- [GGUF naming, conversion, and quantization](docs/distribution.md)
- [Instrumental adapters and real-audio continuation](docs/realaudio-continuation.md)
- [Numerical and listening validation](docs/validation.md)
- [Pinned upstream references](docs/reference-lock.md)

## Models and licenses

This repository contains engine code only and does not redistribute model
weights. The released YuE2, SheetSage2, and MERT2 checkpoints are licensed under
CC BY-NC 4.0; users must obtain them from their upstream publishers and comply
with their terms. Engine code in this repository is MIT licensed.
