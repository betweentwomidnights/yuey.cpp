# third-party notices

yuey's Windows packages carry these alongside the project's `LICENSE`.
`ci/package-windows.ps1` copies this file and `LICENSE-ggml.txt` into the core
zip.

- **ggml**, the tensor library behind every backend DLL (`ggml.dll`,
  `ggml-base.dll`, `ggml-cpu-*.dll`, `ggml-cuda.dll`, `ggml-vulkan.dll`). MIT
  License, copyright (c) 2023-2026 The ggml authors. the full text ships as
  `LICENSE-ggml.txt`. yuey builds from its `ggml` submodule, and the core zip's
  `BUILD-INFO.json` records the exact commit.
- **dr_flac** (`third_party/dr_libs/dr_flac.h`, compiled into the executables)
  decodes FLAC uploads. it's public domain (Unlicense) or MIT No Attribution,
  your choice. neither asks for a notice, and the text is at the end of the
  header. the pinned release and hash are in `third_party/dr_libs/README.md`.
- **Unicode data.** `src/yue2_unicode_tables.h` is generated from Python's
  `unicodedata` (Unicode 15.0.0) by `tools/generate_yue2_unicode_tables.py`.
  Unicode's data files are used under the Unicode License v3.

the split packages don't include the CUDA runtime. gary4local installs it
separately, from its own release, with NVIDIA's EULA alongside. the standalone
zip does include it, and carries `NVIDIA-CUDA-EULA.txt` next to the DLLs.

model weights aren't in any package. each model's license and download terms
stay with its model repository. see `docs/model-cards`.
