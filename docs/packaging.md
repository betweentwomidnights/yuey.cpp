# Packaging and releases

yuey ships prebuilt Windows packages with every GitHub release. They follow
the native runtime package contract that gary4local installs every GGML
service by:
[`docs/native-runtime-packages.md`](https://github.com/betweentwomidnights/gary-localhost-installer/blob/main/docs/native-runtime-packages.md)
in gary-localhost-installer. That document is the source of truth for the
format. This page covers yuey's side of it: what a release contains, how it is
built, and the checklist for cutting one.

## What a release contains

| Asset | Contents | For |
|---|---|---|
| `yuey-vX.Y.Z-windows-x64-core.zip` | `yue2-server.exe`, `yue2-generate.exe`, `yue2-transcribe.exe`, `yue2.dll`, `ggml.dll`, `ggml-base.dll`, `ggml-cpu-*.dll` (every CPU variant), `LICENSE`, `LICENSE-ggml.txt`, `THIRD_PARTY_NOTICES.md`, `BUILD-INFO.json` | supervisors |
| `yuey-vX.Y.Z-windows-x64-cuda.zip` | `ggml-cuda.dll` | supervisors, NVIDIA |
| `yuey-vX.Y.Z-windows-x64-vulkan.zip` | `ggml-vulkan.dll` | supervisors, AMD and Intel |
| `yuey-vX.Y.Z-windows-x64-standalone.zip` | all of the above, plus the CUDA runtime (`cudart64_12`, `cublas64_12`, `cublasLt64_12`, `NVIDIA-CUDA-EULA.txt`), `models.cmd` and a README | running yuey on its own |
| `SHA256SUMS` | one `sha256  name` line per zip, LF endings | everyone |

A supervisor unpacks core and exactly one backend zip into the same folder.
ggml loads backend DLLs from the executable's folder, so that is all the
backend selection there is. The CUDA backend also needs the CUDA runtime on
`PATH`, which a supervisor installs once for every service. gary4local takes it
from gary-localhost-installer's `runtime-cudart-*` release, not from here.

The standalone zip is for everyone else: unpack it, run `models.cmd`, run
`yue2-server.exe --models-dir models`. It runs on any GPU because ggml loads
whichever backend the machine can use; the CUDA DLL simply fails to load on a
machine without an NVIDIA driver. At 711 MB it is mostly the CUDA runtime.

Every zip from a release is attested:

```bash
gh attestation verify yuey-v0.2.0-windows-x64-core.zip -R betweentwomidnights/yuey.cpp
```

This proves the file was built by `.github/workflows/release.yml` from the
tagged commit.

## How the packages are built

`ci/package-windows.ps1` builds, tests, stages and zips everything. The release
workflow runs it, and so can a developer machine, so a package built by hand is
the package CI would have built. It builds with:

- `GGML_NATIVE=OFF`, `GGML_BACKEND_DL=ON`, `GGML_CPU_ALL_VARIANTS=ON`;
- no CUDA architecture list, so ggml's portable default (Maxwell through
  Blackwell) applies;
- CUDA 12.8.1 and Vulkan SDK 1.4.309.0 in CI, on the pinned `windows-2022`
  runner with Visual Studio 2022.

The script also enforces the release rules:

- It refuses a `-Version` that is not `vX.Y.Z`, or that differs from what the
  built server reports. The version is written once, in `project(VERSION)` in
  `CMakeLists.txt`, and `yue2::version()`, `--version`, `/health` and `/props`
  all report it.
- A GPU backend DLL that leaks into the core zip fails the build.
- The output folder is emptied first, so a release never uploads a stale zip.
- `BUILD-INFO.json` records the service, version, yuey and ggml commits, CUDA
  and Vulkan SDK versions, and build time. `dirty: true` marks a hand-built
  package from a checkout with local changes. CI never produces one.

```powershell
ci\package-windows.ps1 -Version v0.2.0               # what CI runs
ci\package-windows.ps1 -Version v0.2.0 -CudaRuntime  # also the cudart zip, for a local supervisor test
```

With `-CudaRuntime`, `dist\` holds everything gary4local installs. Point
`GARY4LOCAL_NATIVE_PACKAGE_DIR` at it to test an install before a release
exists.

## Cutting a release

1. **Version and notes.** Bump `project(VERSION)` in `CMakeLists.txt`, add the
   release's section to `CHANGELOG.md`, and merge both to `main`.
2. **Dry run.** Build the packages in CI without publishing anything:

   ```bash
   gh workflow run release.yml -R betweentwomidnights/yuey.cpp --ref main
   ```

   With no tag the workflow packages `main` under the version CMake declares,
   and keeps the zips as a workflow artifact for 14 days. Download them and
   install them through gary4local (`GARY4LOCAL_NATIVE_PACKAGE_DIR`), on CUDA
   and on Vulkan.
3. **Publish.** Create the GitHub release `vX.Y.Z` from `main`, with the
   `CHANGELOG.md` section as its notes. Publishing starts the workflow, which
   builds the tag, attests the zips and attaches them with `SHA256SUMS`. If a
   runner fails, rerun it for the same tag; assets are replaced, not
   duplicated:

   ```bash
   gh workflow run release.yml -R betweentwomidnights/yuey.cpp -f tag=vX.Y.Z
   ```

4. **Verify.** Check the attached `SHA256SUMS` and attestations against the
   downloaded zips.
5. **Pin it in gary4local.** In gary-localhost-installer:

   ```bash
   node control-center/src-tauri/scripts/pin_native_release.mjs --service yuey --repo betweentwomidnights/yuey.cpp --tag vX.Y.Z
   ```

   Then test an install from the published URLs, with no override set, before
   a gary4local release ships the pin.

A release is immutable once gary4local pins it: a fix is a new patch version,
never re-uploaded assets under an old tag. The one exception is a rerun that
attaches missing assets, before anything has pinned the release.
