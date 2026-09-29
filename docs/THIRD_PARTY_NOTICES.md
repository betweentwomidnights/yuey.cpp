# Third-party notices

yuey's Windows runtime packages carry these alongside the project's `LICENSE`.
`ci/package-windows.ps1` copies this file and `LICENSE-ggml.txt` into the core
archive.

- **ggml**: the tensor library and every backend DLL (`ggml.dll`,
  `ggml-base.dll`, `ggml-cpu-*.dll`, `ggml-cuda.dll`, `ggml-vulkan.dll`). MIT
  License, copyright (c) 2023-2026 The ggml authors. The full text ships as
  `LICENSE-ggml.txt`. yuey builds from the `ggml` submodule, pinned to the
  commit recorded in the core archive's `BUILD-INFO.json`.
- **dr_flac** (`third_party/dr_libs/dr_flac.h`, compiled into the executables):
  decodes FLAC uploads. Public domain (Unlicense) or MIT No Attribution, at the
  reader's choice; neither asks for a notice, and the text is at the end of the
  header. The pinned release and hash are in `third_party/dr_libs/README.md`.
- **Unicode data**: `src/yue2_unicode_tables.h` is generated from Python's
  `unicodedata` (Unicode 15.0.0) by `tools/generate_yue2_unicode_tables.py`.
  Unicode data files are used under the Unicode License v3.

The CUDA runtime (cudart and cuBLAS) is not in these packages. gary4local
installs it separately, from its own release, with NVIDIA's EULA alongside.

Model weights are not in these packages either. Each model's license and
download terms stay with its model repository; see `docs/model-cards`.
