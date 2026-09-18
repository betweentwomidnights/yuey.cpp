@echo off
setlocal enabledelayedexpansion
rem Download yuey.cpp GGUFs with curl.exe, without Python.
rem Usage: models.cmd [--encoding bf16^|q8_0^|q4_k_m] [--profile core^|transcribe^|full] [--namespace HF_USER] [--out DIR] [--dry-run]

set "ENCODING=q4_k_m"
set "PROFILE=full"
set "NAMESPACE=thepatch"
set "OUT=models"
set "DRY_RUN=0"

:parse
if "%~1"=="" goto parsed
if /I "%~1"=="--encoding"  ( set "ENCODING=%~2" & shift & shift & goto parse )
if /I "%~1"=="--profile"   ( set "PROFILE=%~2" & shift & shift & goto parse )
if /I "%~1"=="--namespace" ( set "NAMESPACE=%~2" & shift & shift & goto parse )
if /I "%~1"=="--out"       ( set "OUT=%~2" & shift & shift & goto parse )
if /I "%~1"=="--dry-run"   ( set "DRY_RUN=1" & shift & goto parse )
if /I "%~1"=="-h" goto help
if /I "%~1"=="--help" goto help
echo unknown option: %~1 1>&2 & exit /b 1

:parsed
if /I "%ENCODING%"=="bf16"  ( set "ENC=BF16" & goto encoding_ok )
if /I "%ENCODING%"=="q8_0"  ( set "ENC=Q8_0" & goto encoding_ok )
if /I "%ENCODING%"=="q4_k_m" ( set "ENC=Q4_K_M" & goto encoding_ok )
echo unpublished encoding: %ENCODING% ^(available: bf16^|q8_0^|q4_k_m^) 1>&2 & exit /b 2
:encoding_ok
if /I "%PROFILE%"=="core" goto profile_ok
if /I "%PROFILE%"=="transcribe" goto profile_ok
if /I "%PROFILE%"=="full" goto profile_ok
echo unknown profile: %PROFILE% ^(core^|transcribe^|full^) 1>&2 & exit /b 2
:profile_ok

set "REPO=%NAMESPACE%/YuE2-3B-GGUF"
if not exist "%OUT%" mkdir "%OUT%"
call :dl "yue2-3.6B-v1.0-%ENC%.gguf"
if errorlevel 1 exit /b 1
call :dl "yue2-vae-v1.0-F16.gguf"
if errorlevel 1 exit /b 1
call :dl "yue2-qwen.tiktoken"
if errorlevel 1 exit /b 1
if /I not "%PROFILE%"=="core" (
  call :dl "sheetsage2-mert2-0.7B-v1.0-F16.gguf"
  if errorlevel 1 exit /b 1
)
if /I "%PROFILE%"=="full" (
  call :dl "yue2-instrumental-cot-full-v1.0-F16-LoRA.gguf"
  if errorlevel 1 exit /b 1
  call :dl "yue2-realaudio-nar-v9-v1.0-F16-LoRA.gguf"
  if errorlevel 1 exit /b 1
  call :dl "yue2-semantic-tokenizer-0.7B-v1.0-F16.gguf"
  if errorlevel 1 exit /b 1
)
echo [done] Yuey %PROFILE% ^(%ENC%^) -^> %OUT%\
exit /b 0

:help
echo Usage: models.cmd [--encoding bf16^|q8_0^|q4_k_m] [--profile core^|transcribe^|full] [--namespace HF_USER] [--out DIR] [--dry-run]
exit /b 0

:dl
set "FILE=%~1"
set "DST=%OUT%\%~1"
set "PART=%DST%.part"
if "%DRY_RUN%"=="1" (
  echo [plan] https://huggingface.co/%REPO%/resolve/main/%FILE% -^> %DST%
  exit /b 0
)
if exist "%DST%" ( echo [skip] %FILE% & exit /b 0 )
if exist "%PART%" ( echo [resume] %FILE% ) else ( echo [download] %REPO%/%FILE% )
curl.exe -fL --retry 3 --continue-at - -o "%PART%" "https://huggingface.co/%REPO%/resolve/main/%FILE%"
if errorlevel 1 exit /b 1
move /Y "%PART%" "%DST%" >nul
exit /b 0
