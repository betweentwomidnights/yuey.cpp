# Yuey training workflow: alignment with SA3 Studio and gary4local

The first implementation is now available in the draft
[local training workflow](training.md), with measured
[conditioning optimizations](training-optimization-2026-10-27.md). The original
contract below records the research-stage boundaries. Its previously described
companion/API, durable named runs, device ownership and paired publication now
have an initial implementation; automatic neutral replay, broader musical
validation and downstream Gary integration remain outside that implementation.

The reference is the trainer in `C:/dev/sa3.cpp/web/train.html`, backed by
`tools/sa3-train-web.cpp` and `src/train_web_run.h`, and the native job/catalog
integration described in gary4local's `docs/sa3-cpp-migration.md`. This is local
source inspection, not a proposed change to either neighboring repository.

The full Koan experiment is establishing the numerical recipe and local 8GB
operating limits first. The current Yuey inference presets are experimental
evidence; their hard-coded names are not the desired product interface.

## Shared lifecycle

| Product surface | Required behavior |
| --- | --- |
| New training | User supplies a run/adapter name and dataset folder, chooses an installed compatible base/device, and begins with measured defaults. |
| Data preparation | Source audio/TXT files remain read-only. Show per-track coverage, split membership, caption/score review flags and cache progress before fitting. |
| Run history | Stable run ID, immutable resolved configuration, stage/status, logs, metrics and checkpoint history survive process/app restarts. |
| Stop/resume | Request a checkpoint after the current update; retain process/GPU ownership until exit. Resume the exact model, data, adapter family, preprocessing and sampler state. |
| Model registration | Publish validated inference artifacts into a managed catalog without changing optimizer checkpoints. Preserve old adapters and reject duplicate user names. |
| Inference | Refresh the existing dropdown from the catalog; use the user's adapter name and recorded compatible expert pair rather than names encoded in HTML. |
| Prompt pool | Derive a selectable/dice prompt pool from actual training captions. A single hard-coded Koan prompt is a development convenience. |
| Resource coordination | Training, preprocessing, generation and transcription share one device reservation. A running child remains owned through exit and registration. |

SA3's server keeps the training companion separate and proxies `/training/*`
under the Studio origin. Its named runs publish a GGUF, prompt pool and registry
sidecar. Gary similarly retains immutable run/checkpoint files, registers native
exports, and reserves the shared runtime through cancellation and publication.
Those boundaries are useful here even if Yuey's exact implementation differs.

## YuE2-specific boundaries

One user-visible style can contain an AR adapter, a matched NAR adapter, and a
required frozen instrumental companion. These are not three interchangeable
files. The catalog must record base fingerprints, tokenizer-head/preprocessing
identity, AR/NAR source hashes, family/rank/scaling, required companion order and
strength, and DoRA normalization-reference compatibility. Exports should retain
these constraints across a downstream gary4local installation.

Planning stays base/common for this experiment. Direct planning loss, actual
symbolic-off training, semantic tokenizer changes, and expert-pair swapping are
separate capabilities. They should not appear as ordinary generation toggles
before being implemented and validated.

Shared workflow does not imply copying SA3's numeric defaults: its latent RMS,
crop duration and adapter-rank choices are architecture-specific. Yuey's
settings must follow its own measured resource and listening results.

Basic training controls should eventually be dataset, name, device/base,
training budget, measured crop/context profile and checkpoint cadence. Rank,
family, expert learning rates, teacher KL/replay, validation policy and sampler
belong in advanced controls after the full comparison establishes useful
defaults. Head/precision identity, score quantization and vocabulary/EOS policy
should be pinned and recorded; exposing every development option is not a
substitute for a supported workflow.

For inference, retain a small surface: named style, strength, prompt/trigger,
seed and ordinary sampling controls. Separate AR/NAR strengths can be advanced
controls; avoid claiming one scalar guarantees melody adherence. A completed
job should supply its trained caption pool automatically.

## Current reusable pieces and remaining work

`tools/prepare_ar_dataset.py` consumes a versioned job JSON, creates independent
semantic caches and scored examples, verifies identities, and emits structured
preparation progress. It is presently a local Python/NumPy/SoundFile orchestrator
around native model tools; production dependency/packaging choices remain open.
ML training itself remains native. `yue2-score` validates and normalizes ABC and
exports MIDI without loading models, supporting independent pitch/timing checks.

The native AR CLI accepts a prepared JSON dataset and track-balanced window
sampling, with a checkpointed group/window permutation. The manifest avoids
Windows argument-length limits. A native content signature binds the actual
loaded prefix/codec streams, while the job manifest records stronger SHA-256
provenance. A stop-file requests save-and-stop after an update. Existing explicit
sample arguments and historical sampler/state behavior remain supported.

The full-run preparation currently waits for inference to be idle at GPU phase
boundaries. This is cooperative scheduling, **not** the production-wide device
reservation described above. A user-facing training companion/API, durable run
registry, crash recovery and automatic catalog/prompt publication are still
future work. They are not claimed implemented by the research scripts.

NAR now has the corresponding prepared-manifest and named-run helpers:
`prepare_nar_dataset.py` binds exact prefixes/codes/VAE targets to one frozen AR
checkpoint, and `train_nar_run.py` records identities, heldout selection and
explicit resumed segments. On-demand cache loading bounds host memory; native
grouped sampling and stop files preserve optimizer/data-order state. These local
helpers still require a durable job owner and catalog/API before becoming the
user-facing workflow described above.

No SA3 or gary4local files are modified by this contract. The current work
preserves their conventions while gathering the data needed to choose Yuey's
supported defaults.
