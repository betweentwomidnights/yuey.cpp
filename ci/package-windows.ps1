# ABOUTME: Builds yuey's Windows packages: split core/backend zips for a supervisor
# ABOUTME: such as gary4local, one standalone zip for everyone else, and SHA256SUMS.
#
# The backends are built as dynamic libraries (GGML_BACKEND_DL) and the CPU
# backend in every instruction-set variant (GGML_CPU_ALL_VARIANTS), so one core
# package runs on any x64 machine and a GPU backend is a single DLL unpacked
# beside it. ggml looks for backend DLLs next to the executable, which is what
# makes the split work: unpack core, unpack one backend over it, run.
#
# GGML_NATIVE is off so nothing is tuned to the machine that built it, and no
# CUDA architecture list is passed: with native off, ggml's own default covers
# Maxwell through Blackwell as PTX plus real code for the common cards.
#
# Two audiences, two shapes. A supervisor takes the split packages: core, plus
# the one backend zip the machine needs, plus the CUDA runtime (cudart and
# cuBLAS), which is most of a CUDA download, the same for every native service,
# and installed once by gary4local from its own release (gary-localhost-
# installer's runtime-cudart-*), so it is not a split yuey package. Someone
# running yuey on its own takes the standalone zip instead: everything in one
# folder, both GPU backends and the CUDA runtime included, working on any GPU
# because ggml loads whichever backend the machine can run. -CudaRuntime also
# writes the runtime as its own zip, for testing a supervisor install from a
# local folder before any release exists.
#
# The package contract, which every native service's packages follow, is
# docs/native-runtime-packages.md in gary-localhost-installer; docs/packaging.md
# here covers yuey's side of it.
#
# The same script runs in .github/workflows/release.yml and on a developer
# machine, so a package built by hand is the package CI would have built.
#
# Usage:
#   ci\package-windows.ps1 -Version v0.2.0 [-BuildDir build-dist] [-OutDir dist] [-SkipTests] [-CudaRuntime]
param(
    [Parameter(Mandatory = $true)][string]$Version,
    [string]$BuildDir = "build-dist",
    [string]$OutDir = "dist",
    [switch]$SkipTests,
    [switch]$CudaRuntime
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

$root = Split-Path -Parent $PSScriptRoot
Set-Location $root

# The script clears its staging directory before each run. Keep that deletion
# strictly inside this checkout even if a caller passes an absolute path.
$rootPrefix = [System.IO.Path]::GetFullPath($root).TrimEnd('\') + '\'
foreach ($dir in $BuildDir, $OutDir) {
    $full = [System.IO.Path]::GetFullPath($(if ([System.IO.Path]::IsPathRooted($dir)) { $dir } else { Join-Path $root $dir }))
    if (-not $full.StartsWith($rootPrefix, [System.StringComparison]::OrdinalIgnoreCase)) {
        throw "package-windows: $dir must be inside $root"
    }
}
if ($Version -notmatch '^v\d+\.\d+\.\d+$') {
    throw "package-windows: -Version must look like v0.2.0, not $Version"
}

function Fail([string]$message) {
    Write-Error "package-windows: $message"
    exit 1
}

function Invoke-Checked([string]$program, [string[]]$arguments) {
    & $program @arguments
    if ($LASTEXITCODE -ne 0) {
        Fail "$program exited with $LASTEXITCODE"
    }
}

# --- toolchain --------------------------------------------------------------

$cmake = (Get-Command cmake.exe -ErrorAction SilentlyContinue)
if ($cmake) {
    $cmake = $cmake.Source
    $ctest = Join-Path (Split-Path -Parent $cmake) "ctest.exe"
} else {
    foreach ($edition in "Community", "Professional", "Enterprise", "BuildTools") {
        $candidate = "C:\Program Files\Microsoft Visual Studio\2022\$edition\Common7\IDE\CommonExtensions\Microsoft\CMake\CMake\bin\cmake.exe"
        if (Test-Path $candidate) {
            $cmake = $candidate
            $ctest = Join-Path (Split-Path -Parent $candidate) "ctest.exe"
            break
        }
    }
}
if (-not $cmake) { Fail "CMake was not found on PATH or in Visual Studio 2022" }
if (-not $env:CUDA_PATH) { Fail "CUDA_PATH is not set; install the CUDA Toolkit" }
if (-not $env:VULKAN_SDK) { Fail "VULKAN_SDK is not set; install the Vulkan SDK" }

# version.json ships with a full toolkit install; a trimmed CI install may only
# have the versioned folder, which is named for the same release (v12.8).
$versionJson = Join-Path $env:CUDA_PATH "version.json"
if (Test-Path $versionJson) {
    $cudaVersion = (Get-Content $versionJson -Raw | ConvertFrom-Json).cuda.version
} else {
    $cudaVersion = (Split-Path -Leaf $env:CUDA_PATH).TrimStart("v")
}
if ($cudaVersion -notmatch "^\d+\.\d+") { Fail "cannot tell the CUDA version from $env:CUDA_PATH" }
$cudaMajorMinor = ($cudaVersion -split "\.")[0..1] -join "."
$cudaMajor = ($cudaVersion -split "\.")[0]

Write-Host "yue2.cpp   $(git rev-parse --short HEAD)"
Write-Host "ggml       $(git -C ggml rev-parse HEAD)"
Write-Host "version    $Version"
Write-Host "cuda       $cudaVersion ($env:CUDA_PATH)"
Write-Host "vulkan     $env:VULKAN_SDK"

# --- build ------------------------------------------------------------------

Invoke-Checked $cmake @(
    "-S", ".", "-B", $BuildDir,
    "-G", "Visual Studio 17 2022", "-A", "x64",
    "-DCMAKE_BUILD_TYPE=Release",
    "-DGGML_NATIVE=OFF",
    "-DGGML_BACKEND_DL=ON",
    "-DGGML_CPU_ALL_VARIANTS=ON",
    "-DGGML_METAL=OFF",
    "-DYUE2_CUDA=ON",
    "-DYUE2_VULKAN=ON",
    "-DYUE2_BUILD_TOOLS=ON",
    "-DYUE2_BUILD_TESTS=ON",
    "-DBUILD_TESTING=ON"
)
Invoke-Checked $cmake @("--build", $BuildDir, "--config", "Release", "--parallel")

# The fixture-free tests run on the CPU backend, which here is loaded
# dynamically exactly as it will be on a user's machine.
if (-not $SkipTests) {
    Invoke-Checked $ctest @("--test-dir", $BuildDir, "-C", "Release", "--output-on-failure")
}

# The tag names the release, and CMake's project(VERSION) is what the engine
# reports as its version. A package whose server says another version would
# never match what a supervisor pinned.
$bin = Join-Path $BuildDir "bin\Release"
$reported = (& (Join-Path $bin "yue2-server.exe") --version).Trim()
if ("v$reported" -ne $Version) {
    Fail "the build reports version $reported but the release is $Version; bump project(VERSION) in CMakeLists.txt"
}

# --- stage ------------------------------------------------------------------

$stage = Join-Path $BuildDir "package"
if (Test-Path $stage) { Remove-Item -Recurse -Force $stage }
New-Item -ItemType Directory -Force $OutDir | Out-Null

function Stage([string]$name, [string[]]$patterns, [string]$from) {
    $dir = Join-Path $stage $name
    New-Item -ItemType Directory -Force $dir | Out-Null
    foreach ($pattern in $patterns) {
        $found = @(Get-ChildItem -Path $from -Filter $pattern -File)
        if ($found.Count -eq 0) { Fail "$name package: nothing matches $pattern in $from" }
        foreach ($file in $found) { Copy-Item $file.FullName $dir }
    }
    return $dir
}

$coreDir = Stage "core" @(
    "yue2-server.exe",
    "yue2-generate.exe",
    "yue2-transcribe.exe",
    "yue2.dll",
    "ggml.dll",
    "ggml-base.dll",
    "ggml-cpu-*.dll"
) $bin
Copy-Item (Join-Path $root "LICENSE") $coreDir
Copy-Item (Join-Path $root "ggml\LICENSE") (Join-Path $coreDir "LICENSE-ggml.txt")
Copy-Item (Join-Path $root "docs\THIRD_PARTY_NOTICES.md") $coreDir

# Where the package came from, so an installed runtime can be traced back to
# its commits and toolchain without the release page.
$buildInfo = [ordered]@{
    service     = "yuey"
    version     = $Version
    commit      = (git rev-parse HEAD).Trim()
    # A hand-built package from a checkout with local edits is not that
    # commit; say so. CI always builds a clean checkout of the tag.
    dirty       = [bool](git status --porcelain --untracked-files=no)
    ggml_commit = (git -C ggml rev-parse HEAD).Trim()
    platform    = "windows-x64"
    backends    = @("cuda", "vulkan")
    cuda        = $cudaVersion
    vulkan_sdk  = (Split-Path -Leaf $env:VULKAN_SDK)
    built_utc   = (Get-Date).ToUniversalTime().ToString("yyyy-MM-ddTHH:mm:ssZ")
}
[System.IO.File]::WriteAllText(
    (Join-Path $coreDir "BUILD-INFO.json"),
    ($buildInfo | ConvertTo-Json) + "`n",
    (New-Object System.Text.UTF8Encoding($false)))

# A GPU backend that landed in the core zip would load on every machine, and
# one missing from its own zip would never load anywhere. Check both ways.
foreach ($backend in "cuda", "vulkan") {
    if (Test-Path (Join-Path $coreDir "ggml-$backend.dll")) { Fail "ggml-$backend.dll leaked into the core package" }
}
$cudaDir = Stage "cuda" @("ggml-cuda.dll") $bin
$vulkanDir = Stage "vulkan" @("ggml-vulkan.dll") $bin

$cudartDir = Stage "cudart" @(
    "cudart64_$cudaMajor.dll",
    "cublas64_$cudaMajor.dll",
    "cublasLt64_$cudaMajor.dll"
) (Join-Path $env:CUDA_PATH "bin")
Copy-Item (Join-Path $env:CUDA_PATH "EULA.txt") (Join-Path $cudartDir "NVIDIA-CUDA-EULA.txt")

# Everything in one folder, for running yuey without a supervisor.
$standaloneDir = Join-Path $stage "standalone"
New-Item -ItemType Directory -Force $standaloneDir | Out-Null
foreach ($part in $coreDir, $cudaDir, $vulkanDir, $cudartDir) {
    Get-ChildItem -Path $part -File | Copy-Item -Destination $standaloneDir
}
# How to start without a supervisor, and the curl-only model downloader, so
# the zip is all anyone needs.
Copy-Item (Join-Path $root "ci\standalone-README.txt") (Join-Path $standaloneDir "README.txt")
Copy-Item (Join-Path $root "models.cmd") $standaloneDir

# --- zip and checksum -------------------------------------------------------

Add-Type -AssemblyName System.IO.Compression.FileSystem

$archives = [ordered]@{
    "yuey-$Version-windows-x64-core.zip"       = $coreDir
    "yuey-$Version-windows-x64-cuda.zip"       = $cudaDir
    "yuey-$Version-windows-x64-vulkan.zip"     = $vulkanDir
    "yuey-$Version-windows-x64-standalone.zip" = $standaloneDir
}
if ($CudaRuntime) { $archives["cudart-$cudaMajorMinor-windows-x64.zip"] = $cudartDir }

# A release's folder holds exactly its own packages: a zip left from an
# earlier version, or from -CudaRuntime, must not be uploaded beside them.
Get-ChildItem -Path $OutDir -File | Remove-Item -Force

$sums = New-Object System.Text.StringBuilder
foreach ($entry in $archives.GetEnumerator()) {
    $zip = Join-Path (Resolve-Path $OutDir) $entry.Key
    if (Test-Path $zip) { Remove-Item -Force $zip }
    [System.IO.Compression.ZipFile]::CreateFromDirectory(
        (Resolve-Path $entry.Value).Path, $zip,
        [System.IO.Compression.CompressionLevel]::Optimal, $false)
    $hash = (Get-FileHash -Algorithm SHA256 $zip).Hash.ToLowerInvariant()
    [void]$sums.Append("$hash  $($entry.Key)`n")
    $megabytes = [math]::Round((Get-Item $zip).Length / 1MB, 1)
    Write-Host ("{0,-44} {1,8} MB  {2}" -f $entry.Key, $megabytes, $hash)
}

# LF endings and no BOM, so `sha256sum -c SHA256SUMS` works as-is.
[System.IO.File]::WriteAllText(
    (Join-Path (Resolve-Path $OutDir) "SHA256SUMS"),
    $sums.ToString(),
    (New-Object System.Text.UTF8Encoding($false)))

Write-Host "packages -> $OutDir"
