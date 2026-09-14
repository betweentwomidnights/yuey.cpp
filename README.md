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
- Text-to-music, score-conditioned generation, covers, and cover-then-continue
  workflows through the native YuE2 AR, flow, and VAE pipeline.
- Typed BPM, key, and meter controls that are applied to the planning score
  instead of being left to prompt interpretation.
- Score-first length and ending controls that fit generated plans to musical
  bars and let semantic generation reach the score's natural end.
- Best-effort instrumental generation using score-aligned empty lyric sections
  and a duration-preserving rested Vocal lane.
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

Until the public GGUF download is available, place a complete tier under
`models/`. The Q4_K_M laptop layout is:

```text
models/
  YuE2-3B-GGUF/
    yue2-3.6B-v1.0-Q4_K_M.gguf
    yue2-vae-v1.0-F16.gguf
    sidecars/yue2-qwen.tiktoken
  sheetsage2-mert2-0.7B-v1.0-F16.gguf  # needed for transcription/covers
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

```bash
cmake -S . -B build-metal -DYUE2_METAL=ON -DCMAKE_BUILD_TYPE=Release
cmake --build build-metal --parallel
ctest --test-dir build-metal --output-on-failure
```

Copy the same model directory to the Mac before testing
`build-metal/bin/yue2-server`. Backend options are `YUE2_CUDA`, `YUE2_VULKAN`,
`YUE2_METAL`, and `YUE2_HIP`.

## TODO

- [ ] Publish the GGUFs using the same convention as sa3.cpp: one Hugging Face
  repository per model family containing all required components and quant
  tiers, with model cards, source provenance, checksums, and a clear “download
  one complete tier” table.
- [ ] Add `models.sh`, `models.cmd`, and a cross-platform downloader so a fresh
  checkout can install a complete tier directly into `models/`.
- [ ] Fully validate the Vulkan and Metal backends.
- [ ] Produce useful benchmarks on hardware beyond our RTX 5070 Laptop GPU and
  DGX Spark.
- [ ] Validate the C ABI from a standalone iPlug2 project or Ableton extension.

## Documentation

- [Architecture](docs/architecture.md)
- [Server and web API](docs/server.md)
- [C/C++ embedding](docs/embedding.md)
- [GGUF naming, conversion, and quantization](docs/distribution.md)
- [Numerical and listening validation](docs/validation.md)
- [Pinned upstream references](docs/reference-lock.md)

## Models and licenses

This repository contains engine code only and does not redistribute model
weights. The released YuE2, SheetSage2, and MERT2 checkpoints are licensed under
CC BY-NC 4.0; users must obtain them from their upstream publishers and comply
with their terms. Engine code in this repository is MIT licensed.
