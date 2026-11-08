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
#   ci\package-windows.ps1 -Version v0.2.0 -PrebuiltCuda ggml-cuda-bundle
#   ci\package-windows.ps1 -GgmlCudaOnly [-BuildDir build-cuda] [-OutDir ggml-cuda-bundle]
#   ci\package-windows.ps1 -PrintGgmlCudaKey
#
# Compiling ggml-cuda.dll is over 90% of a package build. -GgmlCudaOnly builds
# just that DLL, with the CUDA runtime beside it, and -PrebuiltCuda packages
# such a bundle instead of compiling one, so a release can ship the build its
# dry run tested. Both use this script's one configure line, so the bundle is
# built exactly as a full package build would build it. -PrintGgmlCudaKey names
# everything the DLL depends on; release.yml stores bundles under that name.
param(
    [string]$Version = "",
    [string]$BuildDir = "build-dist",
    [string]$OutDir = "dist",
    [switch]$SkipTests,
    [switch]$CudaRuntime,
    [string]$PrebuiltCuda = "",
    [switch]$GgmlCudaOnly,
    [switch]$PrintGgmlCudaKey
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
$packaging = -not $GgmlCudaOnly -and -not $PrintGgmlCudaKey
if ($packaging -and $Version -notmatch '^v\d+\.\d+\.\d+$') {
    throw "package-windows: -Version must look like v0.2.0, not $Version"
}
if (@(@($PrebuiltCuda, $GgmlCudaOnly, $PrintGgmlCudaKey) | Where-Object { $_ }).Count -gt 1) {
    throw "package-windows: use at most one of -PrebuiltCuda, -GgmlCudaOnly and -PrintGgmlCudaKey"
}
if ($CudaRuntime -and -not $packaging) { throw "package-windows: -CudaRuntime only applies to a package build" }

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

# The Visual C++ runtime, carried app-local. Every binary links it
# dynamically, OpenMP (vcomp140) included. Windows looks in an executable's
# own folder before System32, so copies beside it win over whatever the
# machine has, and nobody needs an administrator install of the
# redistributable. They come from the redistributable folder of the Visual
# Studio that CMake built the package with, and must be at least as new as its
# compiler: binaries from a newer toolset can crash on an older msvcp140
# (std::mutex changed in 14.40).
$vcRuntimeNames = "msvcp140.dll", "vcruntime140.dll", "vcruntime140_1.dll", "vcomp140.dll"

function Get-VcRuntime([string]$buildDir) {
    # Everything comes from the build's own configuration, not from whichever
    # Visual Studio or CMake happens to be newest: the cache names the CMake
    # that configured it and the Visual Studio instance it used, and that
    # CMake's own version folder holds the compiler it found.
    $cache = Get-Content (Join-Path $buildDir "CMakeCache.txt")
    function Cached([string]$name) {
        $line = $cache | Where-Object { $_ -match "^$([regex]::Escape($name)):[A-Z]+=(.+)$" } | Select-Object -First 1
        if (-not $line) { Fail "$buildDir\CMakeCache.txt has no $name" }
        return ($line -replace "^[^=]+=", "")
    }
    $instance = [System.IO.Path]::GetFullPath((Cached "CMAKE_GENERATOR_INSTANCE"))
    $cmakeVersion = (& (Cached "CMAKE_COMMAND") -E capabilities | ConvertFrom-Json).version.string
    $compilerFile = Join-Path $buildDir "CMakeFiles\$cmakeVersion\CMakeCXXCompiler.cmake"
    if (-not (Test-Path $compilerFile)) { Fail "$compilerFile is missing; configure $buildDir with CMake $cmakeVersion" }
    $record = Get-Content $compilerFile -Raw
    if ($record -notmatch 'set\(CMAKE_CXX_COMPILER "([^"]+)"\)') { Fail "$compilerFile names no C++ compiler" }
    $compiler = [System.IO.Path]::GetFullPath($Matches[1])
    if (-not $compiler.StartsWith($instance.TrimEnd('\') + '\', [System.StringComparison]::OrdinalIgnoreCase)) {
        Fail "the compiler $compiler is not part of the Visual Studio at $instance"
    }
    # The compiler reports 19.MM; its runtime is 14.MM.
    if ($record -notmatch 'set\(CMAKE_CXX_COMPILER_VERSION "19\.(\d+)') { Fail "$compilerFile has no MSVC version" }
    $toolsetMinor = [int]$Matches[1]

    $versionFile = Join-Path $instance "VC\Auxiliary\Build\Microsoft.VCRedistVersion.default.txt"
    if (-not (Test-Path $versionFile)) { Fail "$instance has no VC redistributable version file" }
    $redist = Join-Path $instance "VC\Redist\MSVC\$((Get-Content $versionFile -Raw).Trim())\x64"

    $files = @()
    foreach ($name in $vcRuntimeNames) {
        $folder = if ($name -eq "vcomp140.dll") { "Microsoft.VC143.OpenMP" } else { "Microsoft.VC143.CRT" }
        $file = Join-Path $redist "$folder\$name"
        if (-not (Test-Path $file)) { Fail "$file is missing; install the MSVC v143 redistributable files" }
        $info = (Get-Item $file).VersionInfo
        if ($info.FileMajorPart -ne 14 -or $info.FileMinorPart -lt $toolsetMinor) {
            Fail "$name $($info.FileVersion) is older than the 14.$toolsetMinor compiler that built the package"
        }
        $files += $file
    }
    return $files
}

# Records every DLL a process loads, by the path Windows actually opened, by
# running it to exit as its debugger. Unlike listing a live process's modules
# it can't miss a short-lived one, and it sees backends loaded at run time.
if (-not ("DllLoadRecorder" -as [type])) { Add-Type -TypeDefinition @'
using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Runtime.InteropServices;
using System.Text;

public static class DllLoadRecorder
{
    [StructLayout(LayoutKind.Sequential, CharSet = CharSet.Unicode)]
    struct STARTUPINFO
    {
        public int cb;
        public string lpReserved, lpDesktop, lpTitle;
        public int dwX, dwY, dwXSize, dwYSize, dwXCountChars, dwYCountChars, dwFillAttribute, dwFlags;
        public short wShowWindow, cbReserved2;
        public IntPtr lpReserved2, hStdInput, hStdOutput, hStdError;
    }

    [StructLayout(LayoutKind.Sequential)]
    struct PROCESS_INFORMATION { public IntPtr hProcess, hThread; public int dwProcessId, dwThreadId; }

    [DllImport("kernel32.dll", SetLastError = true, CharSet = CharSet.Unicode)]
    static extern bool CreateProcessW(string application, StringBuilder commandLine, IntPtr processAttributes,
        IntPtr threadAttributes, bool inheritHandles, uint flags, IntPtr environment, string directory,
        ref STARTUPINFO startup, out PROCESS_INFORMATION process);
    [DllImport("kernel32.dll", SetLastError = true)]
    static extern bool WaitForDebugEvent(byte[] debugEvent, uint milliseconds);
    [DllImport("kernel32.dll", SetLastError = true)]
    static extern bool ContinueDebugEvent(int processId, int threadId, uint status);
    [DllImport("kernel32.dll", SetLastError = true)]
    static extern bool TerminateProcess(IntPtr process, uint exitCode);
    [DllImport("kernel32.dll", SetLastError = true)]
    static extern bool CloseHandle(IntPtr handle);
    [DllImport("kernel32.dll", SetLastError = true, CharSet = CharSet.Unicode)]
    static extern uint GetFinalPathNameByHandleW(IntPtr file, StringBuilder path, uint size, uint flags);

    const uint DEBUG_ONLY_THIS_PROCESS = 0x2, CREATE_NO_WINDOW = 0x08000000;
    const uint DBG_CONTINUE = 0x00010002, DBG_EXCEPTION_NOT_HANDLED = 0x80010001;
    const uint EXCEPTION_DEBUG_EVENT = 1, CREATE_PROCESS_DEBUG_EVENT = 3, EXIT_PROCESS_DEBUG_EVENT = 5,
        LOAD_DLL_DEBUG_EVENT = 6, EXCEPTION_BREAKPOINT = 0x80000003;
    // DEBUG_EVENT on x64: code, process id and thread id, then the union at 16.
    const int UNION = 16;

    public static List<string> Run(string exe, string commandLine, string directory, int timeoutMs, out int exitCode)
    {
        var startup = new STARTUPINFO();
        startup.cb = Marshal.SizeOf(startup);
        PROCESS_INFORMATION process;
        if (!CreateProcessW(exe, new StringBuilder(commandLine), IntPtr.Zero, IntPtr.Zero, false,
                DEBUG_ONLY_THIS_PROCESS | CREATE_NO_WINDOW, IntPtr.Zero, directory, ref startup, out process))
            throw new Win32Exception();
        var loaded = new List<string>();
        var debugEvent = new byte[256];
        int deadline = Environment.TickCount + timeoutMs;
        try
        {
            while (true)
            {
                int left = deadline - Environment.TickCount;
                if (left <= 0)
                {
                    TerminateProcess(process.hProcess, 1);
                    throw new TimeoutException(exe + " did not exit within " + (timeoutMs / 1000) + " s");
                }
                if (!WaitForDebugEvent(debugEvent, (uint)left)) continue;
                uint code = BitConverter.ToUInt32(debugEvent, 0);
                int processId = BitConverter.ToInt32(debugEvent, 4), threadId = BitConverter.ToInt32(debugEvent, 8);
                uint status = DBG_CONTINUE;
                if (code == EXCEPTION_DEBUG_EVENT)
                {
                    // The loader's initial breakpoint is ours; anything else is the program's.
                    if (BitConverter.ToUInt32(debugEvent, UNION) != EXCEPTION_BREAKPOINT) status = DBG_EXCEPTION_NOT_HANDLED;
                }
                else if (code == CREATE_PROCESS_DEBUG_EVENT || code == LOAD_DLL_DEBUG_EVENT)
                {
                    // Both start with the image's file handle, which the debugger must close.
                    var file = new IntPtr(BitConverter.ToInt64(debugEvent, UNION));
                    if (file != IntPtr.Zero)
                    {
                        if (code == LOAD_DLL_DEBUG_EVENT)
                        {
                            var path = new StringBuilder(1024);
                            if (GetFinalPathNameByHandleW(file, path, (uint)path.Capacity, 0) > 0)
                                loaded.Add(path.ToString().Replace(@"\\?\", ""));
                        }
                        CloseHandle(file);
                    }
                }
                else if (code == EXIT_PROCESS_DEBUG_EVENT)
                {
                    exitCode = BitConverter.ToInt32(debugEvent, UNION);
                    ContinueDebugEvent(processId, threadId, DBG_CONTINUE);
                    return loaded;
                }
                ContinueDebugEvent(processId, threadId, status);
            }
        }
        finally
        {
            CloseHandle(process.hThread);
            CloseHandle(process.hProcess);
        }
    }
}
'@ }

# Run a staged executable from an unrelated working directory and fail unless
# every Visual C++ runtime DLL it loads, which must be all four (the arguments
# have to load a CPU backend, which brings in vcomp140), came from its own
# folder. The build machine has the runtime in System32 too, so this is what
# proves the package's copies are the ones that load.
function Assert-AppLocalVcRuntime([string]$exe, [string[]]$arguments) {
    $folder = Split-Path -Parent ([System.IO.Path]::GetFullPath($exe))
    $commandLine = (@("`"$exe`"") + ($arguments | ForEach-Object { "`"$_`"" })) -join " "
    $exitCode = 0
    $loaded = [DllLoadRecorder]::Run($exe, $commandLine, $env:TEMP, 120000, [ref]$exitCode)
    if ($exitCode -ne 0) { Fail "$(Split-Path -Leaf $exe) $($arguments -join ' ') exited with $exitCode under the DLL check" }
    foreach ($name in $vcRuntimeNames) {
        $paths = @($loaded | Where-Object { (Split-Path -Leaf $_) -ieq $name })
        if ($paths.Count -eq 0) { Fail "$(Split-Path -Leaf $exe) $($arguments -join ' ') never loaded $name" }
        foreach ($path in $paths) {
            if ((Split-Path -Parent $path) -ne $folder) { Fail "$(Split-Path -Leaf $exe) loaded $name from $path, not from the package" }
        }
    }
    Write-Host "app-local VC runtime: $(Split-Path -Leaf $exe) loads all four from its own folder"
}

# Everything ggml-cuda.dll depends on: the ggml pin, the CUDA toolkit, the
# compiler, how CMakeLists.txt configures ggml (yuey turns CUDA graphs on)
# and how this script configures the build. The project(VERSION) line changes
# every release without touching ggml, so it is left out. Line endings are
# normalized, so a checkout's autocrlf setting doesn't change the name.
function Get-GgmlCudaKey {
    if (-not $env:CUDA_TOOLKIT_VERSION) { Fail "CUDA_TOOLKIT_VERSION is not set (the full toolkit version, like 12.8.1)" }
    $vswhere = "C:\Program Files (x86)\Microsoft Visual Studio\Installer\vswhere.exe"
    if (-not (Test-Path $vswhere)) { Fail "vswhere was not found" }
    $instance = & $vswhere -version "[17.0,18.0)" -latest -property installationPath
    if (-not $instance) { Fail "Visual Studio 2022 was not found" }
    $toolsetFile = Join-Path $instance "VC\Auxiliary\Build\Microsoft.VCToolsVersion.default.txt"
    if (-not (Test-Path $toolsetFile)) { Fail "$toolsetFile is missing" }
    $toolset = (Get-Content $toolsetFile -Raw).Trim()
    $cmakeLists = (Get-Content (Join-Path $root "CMakeLists.txt")) | Where-Object { $_ -notmatch '^project\(yue2_cpp VERSION ' }
    $inputs = (($cmakeLists -join "`n") + "`n" + (Get-Content $PSCommandPath -Raw)).Replace("`r`n", "`n")
    $sha = [System.Security.Cryptography.SHA256]::Create()
    $digest = -join ($sha.ComputeHash([System.Text.Encoding]::UTF8.GetBytes($inputs)) | ForEach-Object { $_.ToString("x2") })
    $ggml = (git -C ggml rev-parse HEAD).Trim()
    return [ordered]@{
        key     = "$($ggml.Substring(0, 12))-cuda$env:CUDA_TOOLKIT_VERSION-msvc$toolset-$($digest.Substring(0, 12))"
        ggml    = $ggml
        toolset = $toolset
    }
}
if ($PrintGgmlCudaKey) {
    Write-Output (Get-GgmlCudaKey).key
    exit 0
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
if (-not $PrebuiltCuda -and -not $env:CUDA_PATH) { Fail "CUDA_PATH is not set; install the CUDA Toolkit" }
if (-not $GgmlCudaOnly -and -not $env:VULKAN_SDK) { Fail "VULKAN_SDK is not set; install the Vulkan SDK" }

if ($PrebuiltCuda) {
    # The bundle must be the one this checkout would build: same pin, toolkit,
    # compiler and configuration, which is exactly what its name records.
    $bundleInfoFile = Join-Path $PrebuiltCuda "BUNDLE-INFO.json"
    if (-not (Test-Path $bundleInfoFile)) { Fail "$bundleInfoFile is missing" }
    $bundle = Get-Content $bundleInfoFile -Raw | ConvertFrom-Json
    $expected = (Get-GgmlCudaKey).key
    if ($bundle.key -ne $expected) { Fail "the ggml-cuda bundle is $($bundle.key), but this checkout builds $expected" }
    $cudaVersion = $bundle.cuda
} else {
    # version.json ships with a full toolkit install; a trimmed CI install may only
    # have the versioned folder, which is named for the same release (v12.8).
    $versionJson = Join-Path $env:CUDA_PATH "version.json"
    if (Test-Path $versionJson) {
        $cudaVersion = (Get-Content $versionJson -Raw | ConvertFrom-Json).cuda.version
    } else {
        $cudaVersion = (Split-Path -Leaf $env:CUDA_PATH).TrimStart("v")
    }
    if ($cudaVersion -notmatch "^\d+\.\d+") { Fail "cannot tell the CUDA version from $env:CUDA_PATH" }
}
$cudaMajorMinor = ($cudaVersion -split "\.")[0..1] -join "."
$cudaMajor = ($cudaVersion -split "\.")[0]

Write-Host "yue2.cpp   $(git rev-parse --short HEAD)"
Write-Host "ggml       $(git -C ggml rev-parse HEAD)"
Write-Host "version    $(if ($packaging) { $Version } else { 'ggml-cuda only' })"
Write-Host "cuda       $cudaVersion ($(if ($PrebuiltCuda) { "prebuilt in run $($bundle.run_id)" } else { $env:CUDA_PATH }))"
if (-not $GgmlCudaOnly) { Write-Host "vulkan     $env:VULKAN_SDK" }

# --- build ------------------------------------------------------------------

# Load CUDA's MSBuild integration from the toolkit itself. CI caches that
# directory, but not the files the installer registers inside Visual Studio.
# An explicit toolset path therefore also works after a toolkit cache hit.
# Backends load at run time, so with a prebuilt ggml-cuda nothing else in the
# build refers to CUDA, and it is configured off.
Invoke-Checked $cmake (@(
    "-S", ".", "-B", $BuildDir,
    "-G", "Visual Studio 17 2022", "-A", "x64"
) + $(if ($PrebuiltCuda) { @() } else { @("-T", "cuda=$env:CUDA_PATH") }) + @(
    "-DCMAKE_BUILD_TYPE=Release",
    "-DGGML_NATIVE=OFF",
    "-DGGML_BACKEND_DL=ON",
    "-DGGML_CPU_ALL_VARIANTS=ON",
    "-DGGML_METAL=OFF",
    "-DYUE2_CUDA=$(if ($PrebuiltCuda) { 'OFF' } else { 'ON' })",
    "-DYUE2_VULKAN=$(if ($GgmlCudaOnly) { 'OFF' } else { 'ON' })",
    "-DYUE2_BUILD_TOOLS=ON",
    "-DYUE2_BUILD_TESTS=ON",
    "-DBUILD_TESTING=ON"
))
$bin = Join-Path $BuildDir "bin\Release"

if ($GgmlCudaOnly) {
    if (git -C ggml status --porcelain --untracked-files=no) { Fail "ggml has local changes; the bundle would not match its pin" }
    $key = Get-GgmlCudaKey
    if ($cudaVersion -ne $env:CUDA_TOOLKIT_VERSION) { Fail "CUDA_TOOLKIT_VERSION is $env:CUDA_TOOLKIT_VERSION but $env:CUDA_PATH is $cudaVersion" }
    Invoke-Checked $cmake @("--build", $BuildDir, "--config", "Release", "--target", "ggml-cuda", "--parallel")
    if (Test-Path $OutDir) { Remove-Item -Recurse -Force $OutDir }
    New-Item -ItemType Directory -Force $OutDir | Out-Null
    Copy-Item (Join-Path $bin "ggml-cuda.dll") $OutDir
    foreach ($name in "cudart64_$cudaMajor.dll", "cublas64_$cudaMajor.dll", "cublasLt64_$cudaMajor.dll") {
        Copy-Item (Join-Path $env:CUDA_PATH "bin\$name") $OutDir
    }
    Copy-Item (Join-Path $env:CUDA_PATH "EULA.txt") (Join-Path $OutDir "NVIDIA-CUDA-EULA.txt")
    $info = [ordered]@{
        key           = $key.key
        ggml_commit   = $key.ggml
        cuda          = $cudaVersion
        msvc_toolset  = $key.toolset
        source_commit = (git rev-parse HEAD).Trim()
        run_id        = $env:GITHUB_RUN_ID
    }
    [System.IO.File]::WriteAllText(
        (Join-Path (Resolve-Path $OutDir).Path "BUNDLE-INFO.json"),
        ($info | ConvertTo-Json) + "`n",
        (New-Object System.Text.UTF8Encoding($false)))
    Get-ChildItem $OutDir | ForEach-Object { Write-Host ("{0,-28} {1,8:N1} MB" -f $_.Name, ($_.Length / 1MB)) }
    Write-Host "ggml-cuda bundle $($key.key) -> $OutDir"
    exit 0
}

Invoke-Checked $cmake @("--build", $BuildDir, "--config", "Release", "--parallel")
if ($PrebuiltCuda) { Copy-Item (Join-Path $PrebuiltCuda "ggml-cuda.dll") $bin }

# The fixture-free tests run on the CPU backend, which here is loaded
# dynamically exactly as it will be on a user's machine.
if (-not $SkipTests) {
    Invoke-Checked $ctest @("--test-dir", $BuildDir, "-C", "Release", "--output-on-failure")
}

# The tag names the release, and CMake's project(VERSION) is what the engine
# reports as its version. A package whose server says another version would
# never match what a supervisor pinned.
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
    "yue2-ar-train.exe",
    "yue2-train.exe",
    "yue2-train-run.exe",
    "yue2-score.exe",
    "yue2-semantic-tokenize.exe",
    "yue2-vae-encode.exe",
    "yue2.dll",
    "ggml.dll",
    "ggml-base.dll",
    "ggml-cpu-*.dll"
) $bin
Copy-Item (Join-Path $root "LICENSE") $coreDir
Copy-Item (Join-Path $root "ggml\LICENSE") (Join-Path $coreDir "LICENSE-ggml.txt")
Copy-Item (Join-Path $root "docs\THIRD_PARTY_NOTICES.md") $coreDir
$vcRuntime = Get-VcRuntime $BuildDir
foreach ($file in $vcRuntime) { Copy-Item $file $coreDir }
if ($PrebuiltCuda) {
    # The rule Get-VcRuntime applies to everything built here: the runtime
    # shipped must be at least as new as the compiler, including this DLL's.
    $bundleMinor = [int](($bundle.msvc_toolset -split "\.")[1])
    $runtimeMinor = (Get-Item $vcRuntime[0]).VersionInfo.FileMinorPart
    if ($runtimeMinor -lt $bundleMinor) { Fail "the ggml-cuda bundle needs a 14.$bundleMinor VC runtime; this build ships 14.$runtimeMinor" }
}
Assert-AppLocalVcRuntime (Join-Path $coreDir "yue2-server.exe") @("--props")

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
    msvc_runtime = (Get-Item $vcRuntime[0]).VersionInfo.FileVersion
    built_utc   = (Get-Date).ToUniversalTime().ToString("yyyy-MM-ddTHH:mm:ssZ")
}
if ($PrebuiltCuda) {
    # This run's attestation covers the zips; the one on the CUDA backend DLL
    # itself belongs to the run named here.
    $buildInfo.ggml_cuda_built_by_run = $bundle.run_id
    $buildInfo.ggml_cuda_built_from_commit = $bundle.source_commit
    $buildInfo.ggml_cuda_key = $bundle.key
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
) $(if ($PrebuiltCuda) { $PrebuiltCuda } else { Join-Path $env:CUDA_PATH "bin" })
if ($PrebuiltCuda) {
    Copy-Item (Join-Path $PrebuiltCuda "NVIDIA-CUDA-EULA.txt") $cudartDir
} else {
    Copy-Item (Join-Path $env:CUDA_PATH "EULA.txt") (Join-Path $cudartDir "NVIDIA-CUDA-EULA.txt")
}

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
# Training is native: yue2-train-run setup writes the runtime the server's
# --training-config reads, and the server runs yue2-train-run for each run.
Copy-Item (Join-Path $root "docs\training.md") (Join-Path $standaloneDir "TRAINING.md")

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
