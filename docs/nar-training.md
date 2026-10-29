# Experimental native NAR LoRA training

The first native trainer and a bounded Koan Sound pilot are implemented. This
follows the [feasibility investigation](training-history.md#feasibility-investigation-2026-10-22). It is a
CLI and C++ API for experiments, with actual GGML backward execution; PyTorch is
used only for preprocessing and an independent numerical oracle.

The current GGML 0.27 pin has separate
[training migration validation](training-history.md#ggml-027-migration-2026-10-24). The local
validated binaries are in `build-ggml027-cuda/bin/Release`; the original
`build-cuda` binaries are retained for comparison. The pilot measurements and
commands below describe the October 22 run.

The [expanded October 24 pilot](training-history.md#expanded-nar-pilot-2026-10-24) subsequently
prepared 48 windows across 12 training tracks and completed 192 updates, with
all six listening-test tracks kept out of training. Its candidate and blind
comparisons against the original step-48 reference are documented separately.

The user's [clarified style-transfer objective](yue2-style-transfer-objective.md)
is the next evaluation gate. The fidelity gains reported so far do not establish
that unseen-input remixes or standard generations acquire the dataset's style.
The [October 25 style-capability gate](training-history.md#nar-only-style-gate-failed-2026-10-25) now provides
fresh-score, user-audio and standard-generation comparisons with neutral
captions, plus a native-NAR control without v9.

## What is implemented

For expanded scored datasets, the CLI now accepts `--dataset prepared.json`
using format `yue2-nar-prepared-dataset-v1`, `completed: true` and rows with
`cache`, `split` (`train`, `validation`, `retention`) and a stable `group` or
`source_sha256`. Cache paths can be relative to the manifest. This replaces
explicit sample arguments. Add `--shuffle-samples --group-samples` to balance
recordings and exhaust each recording's shuffled windows before reuse.

`--stream-samples` reads frozen conditioning caches on demand, bounding host
memory to one decoded sample per split plus loader transients. The default
explicit-sample mode retains its historical eager loading and numerical/state
compatibility. Manifest runs bind actual cache contents and group assignments
to resume identity. The preparation helper additionally pins SHA-256 identities
for model, AR adapter, numerical runtime, prefixes, codes and VAE targets.
CPU and CUDA conditioning remain distinct cache identities.

`--stop-file PATH` saves after the current update before exiting. A run owner
must retain device/process ownership through exit and must explicitly decide
whether to resume; a successful stop does not imply the training budget is done.
The [full-corpus experiment](training-history.md#full-corpus-koan-training-2026-10-26-to-27) documents the
resource evidence and tests. The listener selected full-data AR DoRA500 and
prefers its expanded NAR LoRA400 calibration. All 197 CPU conditioning caches,
resource/gradient checks, the 400-update fit and fixed-token auditions completed.

`tools/train_nar_run.py` (since replaced by `yue2-train-run fit-nar`; see
[native workflow](training.md#native-workflow)) consumes a `yue2-nar-training-v1` job JSON and records
immutable model/cache/runtime identities, named run status, checkpoint metrics
and recording-balanced heldout selection. It supports explicit resumed segments
and retains child ownership through advisory progress-file failures. The current
native CLI pins weight decay `.01` and clipping `1`; the helper rejects other
values instead of recording settings that were not applied. A CPU integration
check verifies full/resumed/stop-resumed state equality, terminal status and
rejection of changed cache bytes before native launch. This is a local
orchestrator, not the future HTTP trainer or model registry.

`yue2-train` trains functional LoRAs on all seven acoustic projections in all
28 NAR layers: Q/K/V/O and MLP gate/up/down, or 196 targets. AR, norms, timestep
embedding, audio boundary projections, semantic tokenizer and VAE remain frozen.
The existing v9 companion, including its four full boundary replacements, is
preserved as a fixed adapter.

The trainer caches frozen AR K/V, releases the AR model, then loads only NAR
weights. The unquantized BF16 checkpoint is represented in F16 for frozen-weight
backward, with F32 trainable factors, gradients and AdamW state. Quantized
training weights are rejected. Q4 remains usable for checkpoint auditioning.

Forward execution saves block boundaries; backward recomputes each block and
propagates its vector-Jacobian product. Separate forward/backward graphs share
four workspaces by graph shape. The measured 256-frame, rank-8 case reserved
88 MiB of graph workspace. Total observed GPU use was about 3.9 GB, including
background applications, on the RTX 5070 Laptop 8 GB. This is not a measured
upper bound for longer windows.

Training uses batch one, constant learning rate, AdamW, global gradient clipping
and uniformly sampled noise fractions in [0.05, 0.95]. Sampling is with replacement
over prepared windows. Held-out evaluation uses fixed noise and fractions
0.2/0.5/0.8. Checkpoints save adapter factors, both optimizer moments, optimizer
step and the training RNG; identity checks reject changed rank, shape,
hyperparameters, base fingerprint, fixed companion paths or run identity.
Files are committed by temporary-file rename and existing checkpoints are not
silently overwritten.

**Exports contain the learned delta. Continue loading the same fixed v9
companion when using a trained checkpoint.** They do not independently replace
the v9 companion or bundle its weights.

## Validation on the full model

The downloaded BF16 GGUF passed its published SHA-256:
`9e1e3bb89dce35703345cd39c57a43f996d62e5ca87afb6784ff5ecfcad27a55`.
It is available in [the published model repository](https://huggingface.co/thepatch/YuE2-3B-GGUF).

| Check | Result |
|---|---|
| Full 28-layer CUDA known-token forward / PyTorch | 0.293% relative L2 prediction error |
| Full 28-layer CUDA known-token gradients / PyTorch | 0.821% relative L2 gradient error |
| First native AdamW update / PyTorch, using identical gradients | Maximum absolute parameter error 7.45e-9 |
| Full 28-layer CUDA real-audio/v9 forward / PyTorch | 0.329% relative L2 prediction error |
| Full 28-layer CUDA real-audio/v9 gradients / PyTorch | 0.673% relative L2 gradient error |
| Full CUDA gradients after a parameter update | 0.603% relative L2 error; maximum significant factor error 3.22% |
| Full CPU known-token forward / PyTorch | 0.0307% relative L2 prediction error |
| Full CPU known-token gradients / PyTorch | 0.0522% relative L2 gradient error |
| Existing inference velocity / new training velocity | 0.421% relative L2 error |
| Four-track stochastic run: continuous 48 vs resume from 24 | Adapter **and complete optimizer/RNG state files byte-identical** |
| Zero-valued learned adapter | Bit-identical reconstruction to v9 baseline on both held-out passages |
| Source-latent strength 0 / strength 1 | Bit-identical to VAE-only decode / ordinary generation respectively, on both passages |
| Actual ABC remix | AR semantic tokens byte-identical across all five NAR checkpoint variants, on both passages |
| Regression tests | 17/17 native CTests and 19/19 Python tests passed |

The oracle uses PyTorch autograd and block checkpointing, reads frozen tensors
directly from GGUF, and independently implements GQA, RoPE, RMSNorm, SwiGLU,
timestep embedding and flow loss. It checks the complete gradient vector;
significant individual factors must also remain within 5% relative error.
The initial GPU checks' largest individual factor errors were 3.56% for the
known-token case and 1.43% for real-audio/v9. The known-token target was rendered
with the unquantized base from an actual AR-generated token stream, without
predicting tokens back from audio.

The known-token fixed-noise overfit check reduced its training loss from
0.1717 to 0.0407 in 16 steps. Its fresh-noise evaluation did not improve; this
is a correctness/capacity control, not a generalization result.

The October 22 pilot used the CONCAT backward correction from
[shared GGML draft PR #18](https://github.com/betweentwomidnights/ggml/pull/18),
candidate `c7491b1c223799e6026ec7a1963925c97d414fb2`. The shared GGML 0.27 refresh
includes that correction and its independent 36-case CPU/CUDA regression.
The local submodule now selects `b77ae790e628270af46a8af1eca38441c23f1252`, the
same pin proposed by [Yuey PR #34](https://github.com/betweentwomidnights/yuey.cpp/pull/34).
See [the migration validation](training-history.md#ggml-027-migration-2026-10-24) for the separate
training and checkpoint checks at that revision. The measurements above remain
the original pilot results; they are not measurements of the refreshed backend.

## The bounded real-audio pilot

Each track contributes one 256-frame / 10.24-second window beginning at 20
seconds. Full-track semantic extraction precedes cropping. VAE targets retain
50 latent frames / two seconds of source context on each side before their
cores are cropped. Only **40.96 seconds of target audio** were used for the
four-track training run; this does not train on the full catalogs.

| Split | Tracks |
|---|---|
| Training | Cobalt, Drift, Evocation, Prism Pulse |
| Whole-song holdout | Aeon, Silkwave |

Configuration: rank 8, 48 optimizer steps, learning rate 5e-5, weight decay 0.01,
clip norm 1, seed 831001. The v9 companion is fixed throughout.

| Step | Mean training-window evaluation loss | Mean held-out evaluation loss |
|---:|---:|---:|
| 0 | 1.16739 | 1.14443 |
| 8 | 1.16143 | 1.13954 |
| 16 | 1.15384 | 1.13342 |
| 24 | 1.14491 | 1.12681 |
| 32 | 1.13514 | 1.12038 |
| 40 | 1.12711 | 1.11574 |
| 48 | 1.12225 | 1.11311 |

Held-out flow loss improved by about **2.7%**. Matched Q4 inference renders use
the same tokens and noise for each checkpoint, preserving the ordinary runtime
as the audition target.

| Held-out passage | v9 log-spectral L1 | Four-track step 48 | v9 chroma / onset | Step 48 chroma / onset |
|---|---:|---:|---:|---:|
| Aeon | 0.950 | 0.848 | 0.667 / 0.899 | 0.680 / 0.902 |
| Silkwave | 1.027 | 0.864 | 0.659 / 0.781 | 0.662 / 0.798 |

The trained renders are about 0.9–1.0 dB quieter than baseline. Equal-RMS
log-spectral comparison still improves on both passages (0.959 → 0.891 and
1.019 → 0.854), but neither spectral distance nor flow loss establishes faithful
instrument sound. Chroma and onset changes are small. Listening remains the
selection criterion; no checkpoint is promoted as a successful timbre-transfer
model on these measurements alone.

A separate one-track run on Cobalt used rank 8, learning rate 1e-4 and 64
random-noise steps. Its training-window evaluation fell from 1.2279 to 1.0734.
Held-out evaluation reached 1.1156 at step 32, then rose to 1.1294 at step 64.
That run is retained as an overfit control, not selected as the preferred model.

## Source-latent and actual-workflow controls

`yue2-render-tokens --source-latents ... --strength s` starts from
`(1-s)*source + s*noise`, integrates from noise fraction `s` to zero, and uses the
same semantic conditioning. `s=0` decodes the source latents; `s=1` is ordinary
generation. The default `FlowOptions::start_noise_level=1` retains ordinary
flow solving. Callers using a lower start fraction must supply a state already
mixed at that fraction.

At strength 0.35, the held-out controls retain substantially more measured
content than pure-noise v9 reconstruction. This is expected when source
information is retained, and trades transformation freedom for preservation.
The experiment uses the original captions; it does **not** establish successful
transfer to a different requested instrumentation.

The listening set also runs the actual workflow separately: transcribe each
held-out excerpt with SheetSage2, apply the instrumental score workflow,
generate fresh AR semantic tokens, and compare NAR checkpoints with the same
AR instrumental adapter and v9 companion. `yue2-generate --semantic-output`
exports the raw codec stream so its identity can be checked across variants.
No real-audio semantic tokens or source latents enter this second workflow.

Open the local `outputs/nar-training/listening.md` for source, VAE-only,
baseline, checkpoint, equal-RMS and actual-remix comparisons. Original float
WAVs are retained for measurement. Listening copies use a common peak-prevention
gain per comparison group; the optional equal-RMS pair is labeled separately.

Local checkpoints, raw diagnostics, source hashes, prepared caches, history and
audio remain under `outputs/nar-training/`. The pilot checkpoint is
`outputs/nar-training/koan-run/step-48.gguf`; its SHA-256 is
`48ec6d227b8cd4915a908b42958e5918035a18bbb788636c242edee453ed5c24`.

## Building and running

Build `yue2-train`, `yue2-render-tokens`, `yue2-semantic-tokenize`,
`yue2-transcribe` and `yue2-generate` with the existing CMake configuration and
the GGML candidate. No CUDA-enabled PyTorch package is required for native
training. Python helpers need `numpy scipy soundfile gguf torch` for CPU VAE
preprocessing and numerical validation.

The small preprocessing/audition helpers are:

- `prepare_nar_experiment.py features`: explicit training/holdout filenames,
  full-track tokenization, VAE-context extraction and source/caption identities.
- `prepare_nar_experiment.py cache`: immutable frozen-AR caches and an identity
  covering source/model/companion content and serialized cache hashes.
- `check_nar_training_reference.py`: independent full-model forward/backward
  and optional first-AdamW-step comparison.
- `audition_nar_experiment.py`: identical-token checkpoint comparisons,
  source-latent controls, and optional actual transcription/remix checks.
  Use a new `--label` after rebuilding a renderer; recorded provenance prevents
  silently reusing renders from another binary or model.

Example cache and training commands:

```powershell
build-cuda/bin/Release/yue2-train --prepare `
  --model models/YuE2-3B-GGUF/yue2-3.6B-v1.0-BF16.gguf `
  --tokenizer models/YuE2-3B-GGUF/sidecars/yue2-qwen.tiktoken `
  --tokens window.i32 --latents posterior-means.f32 --prompt 'instrumental glitch hop' `
  --fixed-lora models/yue2-realaudio-nar-v9-v1.0-F16-LoRA.gguf `
  --cache-out train-window.gguf --device cpu

build-cuda/bin/Release/yue2-train `
  --model models/YuE2-3B-GGUF/yue2-3.6B-v1.0-BF16.gguf `
  --sample train-window.gguf --validation heldout-window.gguf `
  --fixed-lora models/yue2-realaudio-nar-v9-v1.0-F16-LoRA.gguf `
  --out outputs/my-nar-run --identity HASH_OF_DATA_AND_MODEL_CONTENT `
  --device cuda --rank 8 --lr 0.00005 --steps 48 --save-every 8 --eval-every 8
```

Repeat `--sample` / `--validation` for additional windows of the same frame
count. `--steps` means additional updates when resuming. Resume with the same
configuration and `--resume outputs/my-nar-run/step-24.state.gguf`, choosing
additional checkpoint names that do not already exist. The prepared
`cache-identity.json` supplies the dataset/model identity; the exact pilot
command is saved in `outputs/nar-training/koan-run-command.json`.

To inspect gradients, use `--probe-only`. It writes a diagnostic GGUF and
current adapter without applying an optimizer step. Pass the diagnostic file
and the same base/fixed companion to `check_nar_training_reference.py`.

## Current limits

This is an in-memory, batch-one window trainer with host AdamW. Large-corpus
streaming, random windows across entire songs, accumulation, schedules, EMA,
automatic checkpoint auditioning during a running process, and graceful
interruption handling are not implemented. Periodic committed state files
provide resume. Exact resume was verified on the same CUDA build/backend;
cross-backend or cross-toolchain bit identity is not claimed.

Training has used only short windows. The subsequent October 24 auditions
include 30/60/120-second renders and
[a user listening review](training-history.md#nar-pilot-listening-review-2026-10-24): six clear short-clip
preferences favor the learned step-48 adapter, with two near ties leaning the
same way. A four-clip seed-123 spot-check adds three preferences for step 48
and one shared-artifact case without a preference. The 60/120-second comparisons
were rated as ties. Full-song stability,
broader seed consistency and specific timbre transfer remain unestablished.
The tokenizer/head, AR/planner and decoded-audio loss graph are not trained.

CPU frozen-conditioning preparation and full CPU NAR gradients passed. The
existing CPU cached inference/flow path hit a SET_ROWS index assertion while
preparing the synthetic control. Minting and flow-reference preparation in the
new CLI therefore require CUDA; supplied-latent cache preparation supports CPU.
This limitation does not affect the tested CPU NAR training graph, which uses
functional concatenation rather than in-place cache writes.

For staged AR/NAR experiments, `--conditioning-lora AR.gguf` (repeatable)
identifies frozen AR adapters used when preparing the conditioning cache.
Supply the same list during training; it is recorded and checked in caches
and optimizer checkpoints, without loading AR tensors into the NAR trainer.
`--fixed-lora` continues to mean fixed NAR weights. `--prefix prefix.i32`
prepares exact token conditioning instead of rebuilding it from text/ABC.
`--f32-projections` uses F32 projection arithmetic with frozen F16 storage,
and is guarded on resume. Its default remains off for legacy compatibility.
`--shuffle-samples` visits every sample once per shuffled epoch, saving the
permutation, cursor and independent noise RNG for exact resume.

`--noise-schedule uniform|sigmoid` selects the training noise times. Uniform
retains the legacy [.05,.95] distribution and exact RNG/checkpoint behavior.
Sigmoid maps standard-normal draws through a sigmoid using a separate saved
time RNG. Both arms consume the same legacy uniform draw, preserving example
ordering and Gaussian flow noise for paired comparisons. Resuming with a
different schedule is rejected. Sigmoid cannot use `--fixed-noise`, whose time
is fixed at .5. Evaluation still uses fixed noise at .2/.5/.8 and additionally
reports `heldout_per_sigma` in that order. Keep the schedule in the external run
identity as well as the checkpoint's RNG marker.

The [toolkit follow-up](training-history.md#toolkit-follow-up-and-ar200-2026-10-25) compares both schedules
for 100 matched updates and records the listener's preference for uniform.

`--retention sample.gguf` (repeatable) adds evaluation-only neutral controls.
They use the same fixed evaluation noise and sigma .2/.5/.8, but report separate
`retention_eval`, `retention_per_sample` and `retention_per_sigma` fields.
They do not enter the training sampler or the real heldout selection mean.
Include retention cache fingerprints in the caller's run identity. A fixture
verifies that adding retention evaluation leaves optimizer/RNG states identical,
including after resume.

Further training should follow listening review of the held-out reconstruction
and actual-remix sets. The present results show a working native update path and
small held-out gains; they do not show that the v9 semantic bottleneck has been
removed.

The subsequent [October 25 style-capability gate](training-history.md#nar-only-style-gate-failed-2026-10-25)
failed: the user hears essentially identical primary outputs and no Koan-style
match. Pause scaling this recipe. The [diagnosis](training-history.md#nar-only-style-gate-failed-2026-10-25)
separates proven native-training correctness and fidelity changes from the
unmet style objective, and specifies the next feasibility gates.

The later [AR200 matched calibration](training-history.md#ar200-nar-calibration-2026-10-26)
uses a different semantic adapter and broader scored training coverage. The user
endorses its triplet phrasing and prefers its matched NAR400 acoustic output.
Both the combined AR200/NAR400 and AR200-only exports are now discoverable in
the web UI. This later relative result does not erase the earlier NAR-only
failure or establish lossless semantic reconstruction.
