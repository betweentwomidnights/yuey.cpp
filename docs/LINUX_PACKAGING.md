# Experimental Linux runtime packages

Adapted from the validated sa3.cpp Linux package contract. Each engine owns its
own builder; there is no runtime or build dependency on another engine checkout.
The pinned ggml submodule is preserved. CPU generation is not a validation target:
CPU libraries and model-free tests support the CUDA/Vulkan generation paths.

`ci/package-linux.py` produces flat `yuey-vX.Y.Z-linux-{x64,arm64}-core.zip`
and selected `-cuda.zip` / `-vulkan.zip` overlays, with
`SHA256SUMS-linux-{x64,arm64}`. Extract core and exactly one backend together.
The core carries the existing upstream tools, licenses and data, including
yue2-server. `BUILD-INFO.json` records source/ggml commits, dirty state,
toolchains, actual CUDA code generation and ELF dependency requirements.

## Local build and verification

Build natively with CMake, C/C++ compilers, binutils, Python 3.10+, the pinned
ggml submodule, and the selected GPU development toolkits. The full ARM CPU
support-variant set requires SME-capable compilers (GCC 14 is tested); select
local toolchains with `CC`/`CXX`. Python is a packaging/test dependency only.

```sh
python3 ci/package-linux.py --version v0.3.1 --backends cuda vulkan \
  --cuda-arch native --require-gpu --jobs 4
```

`--cuda-arch native` marks a local GPU smoke build. Omit it for ggml's portable
CUDA architecture policy. `--require-clean` rejects source changes. Build and
output directories must be dedicated, marker-owned directories inside the
checkout, with source directories and unowned data refused. Compiler outputs
are reused; only the packager's staging directory is replaced.

The script runs model-free CTests, rejects non-relocatable ELF search paths,
then checks archives after extracting elsewhere with `LD_LIBRARY_PATH` removed.
Internal libraries must resolve inside the extracted directory. All servers
must report their compiled versions and CPU support devices; `--require-gpu`
also requires the selected GPU backend in their properties. Executable modes
are retained, library aliases are ordinary copies, and checksums are written
only after archive verification. These controls do not prove generation,
training, adapter parity or perceptual quality; validate those separately.

Model weights, CUDA redistributables and GPU drivers are not bundled. The actual
external SONAMEs and minimum glibc/GLIBCXX symbol versions are recorded rather
than assuming compatibility with an older distribution. Windows CUDA packs
cannot be used on Linux. No model downloads are performed by this script.

## CI and publishing

`linux-packaging.yml` packages Vulkan on native Ubuntu 24.04 x64 and ARM64
runners, including extracted checks, with seven-day artifact retention. It also
runs through `release.yml` for a no-tag dispatch with `platforms=linux` or `all`.
Linux publishing stays disabled; published releases and tagged rebuilds retain
their existing Windows/macOS behavior. Hosted jobs do not validate GPU execution.

```sh
gh workflow run release.yml -R betweentwomidnights/yuey.cpp \
  --ref feature/linux-runtime-packaging -f platforms=linux
```

On Spark, use the dedicated `/home/kev/gary4local-packaging/yuey.cpp` checkout.
Keep logs, temporary files, toolchains/caches, tests and optional model copies
under its parent workspace, and update `WORKSPACE.md` / `storage.tsv`. Production
services and their model stores are separate. Check artifact metadata/hashes
before exercising downloaded CI binaries; choose unused loopback ports and
stop private test servers afterwards. Preserve desired evidence before cleanup.
