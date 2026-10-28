# Native GGML VAE target preparation

VAE targets now use the C++ `VaeEncoder` and `yue2-vae-encode` batch tool.
**PyTorch is removed from the production trainer and its dependencies.** Python
coordinates manifests, hashes and process ownership; audio decoding, resampling
and encoder inference run natively. The existing PyTorch encoder is retained
only as an optional developer oracle and is not packaged with the trainer.

The same GGML graph runs on CPU, CUDA and Vulkan. CUDA and Vulkan were tested
on the RTX 5070 Laptop GPU. Metal uses these shared operators without a
CUDA-specific implementation, but still requires testing on Apple hardware.
The training ownership/UI gate for macOS remains separate from encoder support.

## Implementation

Only the folded encoder weights are loaded (126.31 MiB). Convolutions, residual
units and SnakeBeta use F32 GGML operations, and the output is the first 64
posterior channels with no sampling. The public API accepts 48 kHz mono/stereo
PCM and returns time-major F32 means. Mono is duplicated without downmixing or
amplitude normalization.

Default tiles contain 128 latent frames plus 16 frames of context on either
side. Tile starts preserve the 1920-sample stride phase. The halo exceeds the
encoder's receptive field; only tile cores are retained. Actual recording edges,
partial final bins and the native decoder's 64-sample trim remain intact.
Longer recordings therefore increase work rather than peak activation memory.
The measured maximum scratch buffer was 892.9 MiB for both CUDA and Vulkan.

F32 im2col avoids rounding the activations to half precision before every
convolution. The graph uses GGML's natural convolution output layout, avoiding
both an extra transpose and CUDA's reduced-precision small-batch MMF dispatch.
No GGML submodule change was required.

The native WAV/FLAC reader preserves independent channels and rejects nonfinite
audio. A VAE-specific native polyphase FIR matches the previous zero-extended
Kaiser-beta-5 resampling profile. The existing inference/transcription resampler
is unchanged. This avoids introducing a different resampling response while
removing Python audio processing from this stage.

## Validation and measured timings

On a real 30-second Koan window (750 frames), the CPU/PyTorch reference encoder
took 9.281 seconds after model loading. Native CUDA took 1.058 seconds as reported
by the CLI, or 1.359 seconds wall time including process startup and external
memory polling. Native Vulkan reported 1.787 seconds including model/backend
initialization. These are individual-window measurements, not a full-dataset
throughput claim. CUDA's observed **total device memory was 1,180 MiB**.

| Comparison against F32 reference | Relative latent L2 | Maximum absolute error |
| --- | ---: | ---: |
| CUDA, identical 48 kHz input | 6.95e-6 | 2.74e-4 |
| Vulkan, identical 48 kHz input | 7.52e-6 | 3.51e-4 |
| Full native reader/resampler/CUDA preparation vs historical target | 7.03e-6 | 2.91e-4 |

The full comparison includes the original 44.1 kHz recording, actual source
context, native resampling, encoding and core cropping. Native 30- and 60-second
target windows were prepared successfully with the bounded tile profile. This
does not change the separate 60-second **AR training** memory limit.
Passing the old and new 30-second targets through the same native decoder gave
waveform difference RMS 0.000630 against signal RMS 0.2803 (relative L2 0.00225).
The outputs are numerically close, not byte-identical; the decoder's half
precision convolutions amplify small latent rounding changes.

Additional checks cover stereo, mono, non-frame-aligned PCM, the decoder's
64-sample trim, a tile seam, full-versus-tiled inference, nonfinite input,
cancellation and same-instance concurrency. Tiled/full relative differences on
the 145-frame fixture were below 4e-7 on CUDA and Vulkan. Existing mono WAV and
FLAC tests pass; new checks cover stereo sample preservation and an independent
polyphase impulse reference.

`tests/vae_encoder_test.cpp` is a manual real-model oracle test. Optional
`tests/test_native_vae_targets.py` runs production preparation with **all Torch
imports blocked**, verifies byte-identical cache resume/recomputation, and
rejects incompatible profiles. Set `YUE2_VAE_ENCODER_TEST_BINARY` and
`YUE2_VAE_ENCODER_TEST_MODEL` to enable it. Set
`YUE2_VAE_ENCODER_TEST_DEVICE` to select a backend (default `cuda`).
A full small web-training run using native targets completed both AR and NAR
fitting and heldout selection.
It also published the paired adapter. A progress regression verifies that fast
native stages expose their final count even when they finish between UI polls.

Portable [measurement evidence](experiments/yue2-native-vae-2026-10-27.json)
records the results and native runtime identity.

## Cache and deployment behavior

Targets pin VAE/binary/backend-library hashes, device, F32 precision, tile and
context settings, resampling policy and source hashes. Incompatible existing
targets are rejected rather than overwritten or silently mixed. A raw target
missing its committed provenance is preserved and recomputed. Completed legacy
targets remain usable by existing runs without loading PyTorch; no active or
historical dataset has been rewritten during this port.

The workflow uses `device` for encoding by default. A runtime can specify
`vae_device`, `vae_binary` or `vae_core_frames` independently; the selected tool
folder must contain its installed backend libraries. The standalone package now
includes `yue2-vae-encode`, and setup verifies it before preparing a dataset.
The UI identifies the stage as native GGML and shows the selected backend.
