# packaging and releases

every yuey release comes with prebuilt Windows packages. they follow the native
runtime package contract that gary4local installs all of its GGML services by:
[`docs/native-runtime-packages.md`](https://github.com/betweentwomidnights/gary-localhost-installer/blob/main/docs/native-runtime-packages.md)
in gary-localhost-installer. that document is the source of truth for the
format. this page is yuey's side of it: what a release contains, how it's
built, and the checklist for cutting one.

## what a release contains

| asset | contents | for |
|---|---|---|
| `yuey-vX.Y.Z-windows-x64-core.zip` | `yue2-server.exe`, `yue2-generate.exe`, `yue2-transcribe.exe`, `yue2.dll`, `ggml.dll`, `ggml-base.dll`, every `ggml-cpu-*.dll` variant, `LICENSE`, `LICENSE-ggml.txt`, `THIRD_PARTY_NOTICES.md`, `BUILD-INFO.json` | supervisors |
| `yuey-vX.Y.Z-windows-x64-cuda.zip` | `ggml-cuda.dll` | supervisors, NVIDIA |
| `yuey-vX.Y.Z-windows-x64-vulkan.zip` | `ggml-vulkan.dll` | supervisors, AMD and Intel |
| `yuey-vX.Y.Z-windows-x64-standalone.zip` | all of the above, plus the CUDA runtime (`cudart64_12`, `cublas64_12`, `cublasLt64_12`, `NVIDIA-CUDA-EULA.txt`), `models.cmd` and a README | running yuey on its own |
| `SHA256SUMS` | one `sha256  name` line per zip, LF endings | everyone |

a supervisor unpacks core and exactly one backend zip into the same folder.
ggml loads backend DLLs from the executable's folder, so that's all the backend
selection there is. the CUDA backend also needs the CUDA runtime on `PATH`,
which a supervisor installs once for all its services. gary4local gets it from
gary-localhost-installer's `runtime-cudart-*` release, not from here.

the standalone zip is for everyone else. unzip it, run `models.cmd`, then run
`yue2-server.exe --models-dir models`. it works on any GPU because ggml loads
whichever backend the machine can use, and the CUDA DLL just fails to load on a
machine without an NVIDIA driver. it's about 711 MB, and most of that is the
CUDA runtime.

every zip in a release is attested:

```bash
gh attestation verify yuey-v0.2.0-windows-x64-core.zip -R betweentwomidnights/yuey.cpp
```

that proves the file was built by `.github/workflows/release.yml` from the
tagged commit.

## how the packages are built

`ci/package-windows.ps1` builds, tests, stages and zips everything. the release
workflow runs it and so can a developer machine, so a package built by hand is
the package CI would have built. it builds with:

- `GGML_NATIVE=OFF`, `GGML_BACKEND_DL=ON` and `GGML_CPU_ALL_VARIANTS=ON`.
- no CUDA architecture list, so ggml's portable default applies (Maxwell
  through Blackwell).
- in CI, CUDA 12.8.1 and Vulkan SDK 1.4.350.0 (the official LunarG installer) on the pinned `windows-2022`
  runner with Visual Studio 2022.

the script also enforces the release rules:

- it refuses a `-Version` that isn't `vX.Y.Z`, or that differs from what the
  built server reports. the version is written once, in `project(VERSION)` in
  `CMakeLists.txt`, and `yue2::version()`, `--version`, `/health` and `/props`
  all report it.
- a GPU backend DLL that leaks into the core zip fails the build.
- it empties the output folder first, so a release never uploads a stale zip.
- `BUILD-INFO.json` records the service, version, yuey and ggml commits, CUDA
  and Vulkan SDK versions, and build time. `dirty: true` marks a hand-built
  package from a checkout with local changes. CI never makes one.

```powershell
ci\package-windows.ps1 -Version v0.2.0               # what CI runs
ci\package-windows.ps1 -Version v0.2.0 -CudaRuntime  # also the cudart zip, for a local supervisor test
```

with `-CudaRuntime`, `dist\` holds everything gary4local installs. point
`GARY4LOCAL_NATIVE_PACKAGE_DIR` at it to test an install before a release
exists.

## cutting a release

1. **version and notes.** bump `project(VERSION)` in `CMakeLists.txt`, add the
   release's section to `CHANGELOG.md`, and merge both to `main`.
2. **dry run.** build the packages in CI without publishing anything:

   ```bash
   gh workflow run release.yml -R betweentwomidnights/yuey.cpp --ref main
   ```

   with no tag, the workflow packages `main` under the version CMake declares
   and keeps the zips as a workflow artifact for 14 days. download them and
   install them through gary4local (`GARY4LOCAL_NATIVE_PACKAGE_DIR`), on CUDA
   and on Vulkan.
3. **publish.** create the GitHub release `vX.Y.Z` from `main`. publishing
   starts the workflow, which builds the tag, attests the zips and attaches them
   with `SHA256SUMS`. if a runner fails, run it again for the same tag. assets
   get replaced, not duplicated:

   ```bash
   gh workflow run release.yml -R betweentwomidnights/yuey.cpp -f tag=vX.Y.Z
   ```

4. **verify.** check the attached `SHA256SUMS` and attestations against the
   downloaded zips.
5. **pin it in gary4local.** in gary-localhost-installer:

   ```bash
   node control-center/src-tauri/scripts/pin_native_release.mjs --service yuey --repo betweentwomidnights/yuey.cpp --tag vX.Y.Z
   ```

   then test an install from the published URLs, with no override set, before
   a gary4local release ships the pin.

once gary4local pins a release, it doesn't change. a fix is a new patch
version, never new assets under an old tag. the one exception is rerunning the
workflow to attach assets that are missing, before anything has pinned the
release.
