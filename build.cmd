@echo off
rem Build yuey.cpp for one Windows backend and prepare a short-command shell environment.
rem Usage: build.cmd [cpu^|cuda^|vulkan]   (default: cpu)
setlocal EnableExtensions EnableDelayedExpansion

cd /d "%~dp0"
set "BACKEND=%~1"
if "%BACKEND%"=="" set "BACKEND=cpu"

set "CMAKE_EXE="
set "CTEST_EXE="
where cmake.exe >nul 2>nul
if not errorlevel 1 (
    set "CMAKE_EXE=cmake.exe"
    set "CTEST_EXE=ctest.exe"
) else (
    for %%E in (Community Professional Enterprise BuildTools) do (
        set "VSCMAKE=C:\Program Files\Microsoft Visual Studio\2022\%%E\Common7\IDE\CommonExtensions\Microsoft\CMake\CMake\bin"
        if exist "!VSCMAKE!\cmake.exe" if not defined CMAKE_EXE (
            set "CMAKE_EXE=!VSCMAKE!\cmake.exe"
            set "CTEST_EXE=!VSCMAKE!\ctest.exe"
        )
    )
)
if not defined CMAKE_EXE (
    echo [yuey] CMake was not found on PATH or in Visual Studio 2022.
    exit /b 1
)

if /i "%BACKEND%"=="cpu" (
    set "DIR=build"
    set "DEVICE=cpu"
    set "FLAGS=-DYUE2_CUDA=OFF -DYUE2_VULKAN=OFF"
) else if /i "%BACKEND%"=="cuda" (
    set "DIR=build-cuda"
    set "DEVICE=cuda"
    set "FLAGS=-DYUE2_CUDA=ON -DYUE2_VULKAN=OFF -DCMAKE_CUDA_ARCHITECTURES=native"
) else if /i "%BACKEND%"=="vulkan" (
    set "DIR=build-vulkan"
    set "DEVICE=vulkan"
    set "FLAGS=-DYUE2_CUDA=OFF -DYUE2_VULKAN=ON"
    if "%VULKAN_SDK%"=="" echo [yuey] WARNING: VULKAN_SDK is not set.
) else (
    echo [yuey] Unknown backend: %BACKEND%  ^(expected cpu, cuda, or vulkan^)
    exit /b 1
)

echo [yuey] configuring %BACKEND% -^> %DIR%\
"%CMAKE_EXE%" -S . -B "%DIR%" -G "Visual Studio 17 2022" -DCMAKE_BUILD_TYPE=Release %FLAGS%
if errorlevel 1 exit /b 1

echo [yuey] building...
"%CMAKE_EXE%" --build "%DIR%" --config Release --parallel
if errorlevel 1 exit /b 1

echo [yuey] testing...
"%CTEST_EXE%" --test-dir "%DIR%" -C Release --output-on-failure
if errorlevel 1 exit /b 1

set "BIN=%CD%\%DIR%\bin\Release"
> env.cmd echo @set "PATH=%BIN%;%%PATH%%"
>> env.cmd echo @set "YUE2_MODELS_DIR=%CD%\models"
>> env.cmd echo @set "YUE2_DEVICE=%DEVICE%"
>> env.cmd echo @echo [yuey] environment ready: yue2-server or yue2-generate --help
> env.ps1 echo $env:Path = "%BIN%;$env:Path"
>> env.ps1 echo $env:YUE2_MODELS_DIR = "%CD%\models"
>> env.ps1 echo $env:YUE2_DEVICE = "%DEVICE%"
>> env.ps1 echo Write-Host '[yuey] environment ready: yue2-server or yue2-generate --help'

echo [yuey] done -^> %BIN%
echo [yuey] activate this PowerShell:  . .\env.ps1
echo [yuey] then launch the UI:       yue2-server
echo [yuey] or generate:              yue2-generate --encoding q4_k_m --prompt "..." --bars 16 --ending outro --out song.wav
