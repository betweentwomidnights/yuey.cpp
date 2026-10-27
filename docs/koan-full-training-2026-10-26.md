# Full-corpus Koan training experiment, 2026-10-26–27

The full run is complete. Both AR families received 800 updates and select
step 500 by recording-balanced validation. The listener prefers **DoRA500**,
then prefers its expanded **NAR LoRA400** calibration over native NAR. All
selected-checkpoint gradient checks pass. The final pair and AR-only alternative
are available through the existing web UI alongside the older entries, with
the existing default retained.
The working artifacts live under `outputs/koan-full-2026-10-26/`; this report
records preparation and resource results independently of the eventual musical
verdict. Existing inference presets remain available.

## Scope and dataset

The folder currently contains 42 recordings, not the 38 assumed earlier, with
42 TXT sidecars and 87.49 minutes of source audio. Thirty-eight recordings are
at least 60 seconds long; four are approximately 45–50 seconds. The previous
six recording-level holdouts remain held out. The full run uses 36 training
recordings and six validation recordings, with 156 training and 38 validation
excerpts of 30 real seconds each. Bar-aligned final windows can overlap earlier
windows. Unique coverage is 69.22 training minutes and 17.08 validation minutes,
approximately 98.6% of source audio overall. Source audio and sidecars are
read-only; caches, transcriptions and run files are separate.

The folder includes Noisia, synthwave, breakcore, ambient and other material in
addition to the `koan_sound_*` tracks. This run deliberately uses that full
distribution. Sidecar style descriptions vary across recordings. Each real
example uses its track's style caption with `koan_palette` and an instrumental
prefix; BPM and key are removed from the caption and represented by the score.
There is no synthetic caption paraphrasing or random caption dropout in this
experiment.

The semantic labels use the v9 head selected by the earlier listening audit.
Scores are transcribed per excerpt, then re-engraved from absolute note times
and pitches under the sidecar tempo/key. The grid is 12 ticks per quarter,
supporting sixteenths and eighth-note triplets. Explicit accidentals preserve
absolute pitches. Chord inversions are retained. This avoids changing the
meaning of notes by merely replacing an ABC header.

An independent native-MIDI audit checked all 194 examples and 14,248 retained
notes against the quantized transcription events. Pitch and timing agree;
codec slices match their full-recording streams and every audio core contains
30 real seconds. This verifies conversion, not transcription accuracy against
the recording. Two sparse-score excerpts remain included and flagged.

The initial preparation flattened chord inversions. That version is retained
as historical output and was never fitted. The corrected `dataset-v2.json`
and `examples-v2/` supply the final `training-dataset.json`.

## Measured local operating profile

Measurements use the local 8GB GPU, rank 32, BF16 base storage, F32 training
compute, the required frozen instrumental companion and TF32 disabled.
Dedicated/shared allocations are Windows WDDM process counters. A small shared
allocation alone does not prove active paging; the large growth and slowdown
at 60 seconds make it unsuitable as the dedicated-memory default here.

| Family | Audio context | Prefix tokens | Peak dedicated MiB | Peak shared MiB | Seconds per update |
| --- | ---: | ---: | ---: | ---: | --- |
| LoRA | 30 s | 1131 | 6900 | 74 | 3.38–3.40 |
| DoRA | 30 s | 1131 | 7728 | 74 | 3.84–3.85 |
| LoRA | 60 s | 1140 | 7881 | 774 | 42.15–42.22 |
| DoRA | 60 s | 1140 | 7885 | 1762 | 99.64 |

The 30-second probes use the largest prepared prefix. LoRA completed three
60-second updates; DoRA was deliberately checkpointed and stopped after one
when shared-memory use became substantial. This was a successful graceful stop,
not a numerical failure. The raw three-update test is marked incomplete for
that reason. The full corpus uses 30 seconds. DoRA has modest remaining VRAM
headroom, so a different score density or concurrent inference can matter.

These are update timings, excluding startup, teacher preparation and full
validation. The full trainer caches teacher distributions in host RAM; early
LoRA fitting used approximately 15.4 GiB working set and 22.4 GiB private
allocation. This is not evidence of support for a machine with only 16GB host
RAM. The test machine has 64GB.

DoRA export now reuses existing graph scratch space. Optimizer state, inference
export and resumed continuation are byte-identical to the previous CUDA fixture.
No GGML source change was required for this improvement.

## Frozen fit and selection protocol

Both families start with byte-identical A/B factors and the same sample order.
Each receives rank 32, learning rate `2e-5`, teacher KL `0.2`, weight decay
`0.01`, gradient clipping `1`, seed `831001`, all seven projections in 28 AR
blocks and a 600,000-supervised-token budget: 800 updates of 750 frames.
Checkpoints and complete evaluations are written every 100 updates. The run
does not stop early based on a family-specific loss curve.

Sampling is balanced by recording, with shuffled group epochs and shuffled
windows exhausted within each group before reuse. One separately authored
common-model replay example is a training group; two distinct authored-score
controls are retention-only evaluation examples. They do not enter held-out
style selection. The final manifest therefore contains 197 rows: 157 train,
38 validation and two retention.

Select the checkpoint with the lowest recording-balanced held-out semantic
cross-entropy, including step zero. Keep retention loss separate. This objective
does not establish fidelity, palette or score adherence, so matched listening
remains required. Historical losses from different datasets are not comparable.
The renderer separately evaluates the previous AR200 on the new validation set.

Planning remains common/base. There is no direct score/planning loss or
symbolic-off branch in this run. Arbitrary audio crops do not receive false
natural-ending labels. The real-source NAR VAE targets can be prepared on CPU
while AR trains; NAR calibration waits for the selected AR candidate and must
use that candidate's conditioning.

## Reproducibility and checks

The LoRA fit completed all 800 updates. Held-out selection chooses **step 500**
(`68be3cfe32ca8425fa3cff1c386f653b2ceffd9a924305538418cf3bab2ac003`).
Recording-balanced validation CE falls from 5.836158 to 5.678097, then rises to
5.710647 at step 800. Neutral retention CE is 3.259800 initially, 3.316682 at
selection and 3.350377 finally. These are objective losses, not listening
preferences. The selected checkpoint's post-training gradient oracle passes
at global relative L2 `4.856e-5`.

DoRA also selects **step 500**
(`91d0e332bb2c371ba79eed2de601dfac64dd5358db34abb69c744b677d0d55bf`).
Its recording-balanced CE is 5.836158 initially, 5.677981 at selection and
5.710932 finally. Selected neutral retention CE is 3.316671. Its post-training
gradient oracle passes at relative L2 `4.820e-5`. The two families' numerical
curves are nearly tied; their CE difference does not establish a useful family
preference. Both curves worsen after 500 despite decreasing training loss.

The previous AR200 evaluates at 5.756765 recording-balanced semantic CE on
exactly the new validation examples and graph size (neutral retention CE
3.281957). The expanded candidates improve this objective, with slightly more
neutral-control drift. This does not by itself prove better sound design or
score adherence.

The listener chose **“Full-data DoRA500”** from the presented fresh-score and
external-source-score seed-831001 native-NAR previews. That choice binds the
expanded acoustic calibration to the selected AR SHA above. It is not a claim
of superiority across all seeds. The full six-case/two-seed page is
`outputs/koan-full-2026-10-26/auditions/listening.md`; its additional AR200/NAR400
clips are acoustic quality references, not controlled primary AR comparisons.

During DoRA, a Windows reader briefly denied replacement of `progress.json`.
The Python status wrapper exited at update 312, while the original native
process continued fitting. The failure logs and original failed pipeline status
are retained. Recovery attached live Windows process handles to the same PID
68292, verified its executable and full command/identity, and resumed observation
without restarting training. The recovered process exited with code zero;
`full-budget-parity.json` verifies exactly matching 800-update LoRA/DoRA
sample/group/token histories and 600,000 supervised tokens per family before
selection, oracle and audition stages. Memory monitoring has a visible
gap around recovery; it is not a continuous peak bound over that gap.

Atomic JSON publication now retries Windows reader locks while retaining the
old complete file. Advisory progress I/O failures no longer abandon the native
child. A real Windows file-sharing fault test verifies that the visible status
is not truncated and publication succeeds after the reader releases its handle.

The frozen job, prepared streams, native executable/DLL identities and source
snapshots are retained beside each run. Native states include optimizer and
group/window sampler state. The CLI also binds actual loaded streams to resume
identity instead of trusting only a caller-provided label.

Validated before or during fitting:

- All 32 Python tests pass, including the Windows reader-lock regression. This
  includes the native ABC/MIDI
  roundtrip and the subsequent NAR target-join/generated-edge checks.
- Native grouped-sampler tests pass.
- CPU manifest integration: balanced groups, stop after an update, byte-identical
  resumed final state, and rejection of changed actual token content.
- CUDA DoRA export scratch regression: byte-identical state, adapter and resume.
- Full-model LoRA and DoRA initial PyTorch CPU gradient oracles pass (global
  relative L2 approximately `4.18e-5` and `4.16e-5`). Selected checkpoints will
  receive another oracle check.

Local evidence: `score-audit.json`, `cli-checks-v2/checks.json`,
`export-workspace-checks/checks.json`, `resource/sustained/assessment.json`,
`initialization-parity.json`, each family's `initial-oracle.json`, and
`fit-memory.jsonl`. `pipeline.json` and each family's `progress.json` report
live state. Original source snapshots retain the helper loaded for LoRA;
`auxiliary-source-identity.json` documents later log-read/hash-verification
guards without changing the native model, commands, data or optimizer.

## NAR preparation support added during AR fitting

The NAR CLI can now consume a prepared manifest and load frozen K/V caches on
demand (`--stream-samples`), retaining at most one sample per split instead of
the entire corpus. Recording-balanced sampling and update-boundary stop files
use the same lifecycle as AR. Native resume identity includes actual loaded
cache contents and group assignments. Streamed files are also checked for
size/modification-time changes after initial validation; the preparation helper
records SHA-256 hashes.

This code is built into a separate `nar-runtime/` directory and does not replace
the frozen AR runtime. CPU regression verifies byte-identical legacy and
streamed optimizer states/exports, recording-balanced sampling, exact resume
and rejection of changed actual cache content. A native storage test checks
bounded live samples, reloading and detection of files changed while loading.

`tools/prepare_nar_dataset.py` joins exact source/window/frame/split identities
between completed AR examples and VAE means, pins the selected AR/common/base
and numerical runtime, and prepares exact-prefix conditioning caches. It keeps
the manifest incomplete until all rows and hashes pass. It handles real targets
and optional, explicitly sourced generated auxiliary targets. Selected-AR
conditioning was computed consistently on CPU for all 197 examples. The cache
audit verifies every prefix/code/clean tensor, all finite K/V values, the exact
ordered common/DoRA500 conditioning list, and maximum AR context of 1,882 tokens.
CPU conditioning has a distinct identity from CUDA conditioning; the earlier
[DoRA150 calibration](koan-ar-dora-nar-calibration-2026-10-26.md) did not establish
strict numerical equivalence between them. CUDA inference transfer therefore
remains a listening check, in addition to each fit's numerical oracle.

The acoustic recipe used ordinary rank-32 LoRA, F32 projections, LR
`3e-5`, weight decay `.01`, clipping `1`, uniform noise times, seed `831001`,
400 updates of 750 frames (300,000 target frames), and evaluation/checkpointing
every 100 updates. Recording-balanced sampling and validation selection retain
the six real recording holdouts; one independent generated replay trains and
two distinct neutral controls only evaluate retention. Acoustic DoRA did not
win the previous matched listening comparisons, so it is not repeated here.
Before this fit, maximum-prefix resource and maximum/short-padded conditioning
gradient checks passed. Saved AR IDs remain fixed for acoustic auditions.

## Expanded acoustic results and preferred export

| NAR updates | Recording-balanced heldout flow MSE | Neutral retention MSE |
| --- | ---: | ---: |
| 0 | 1.259427 | 0.877204 |
| 100 | 1.075574 | 0.806077 |
| 200 | 1.065071 | 0.803947 |
| 300 | 1.061163 | 0.803930 |
| 400 | **1.059538** | **0.802876** |

The 400-update fit covers 300,000 target frames. Its actual first 400 sample and
group selections match the selected AR run's order. The native fit completed
successfully, with 54.8 minutes inside the fitter/monitor interval and 56.2
minutes for the enclosing helper invocation, including cache verification.
Complete training/validation evaluation at three noise levels contributes
substantially to this wall time. CPU conditioning preparation is an additional
checkpoint-specific cost. A production trainer should show these as distinct
phases and distinguish training-loss reporting from full validation cadence.

The 30-second NAR resource test completed three actual updates at the maximum
context. The full fit observed 4,376 MiB total GPU allocation; WDDM sampling
recorded approximately 4,376 MiB dedicated and 74 MiB shared. The streamed
fit's sampled host working set peaked at 2.31 GiB and private allocation at
6.84 GiB. These NAR numbers do not replace the larger AR host-memory figures.

Independent full-model oracle relative errors:

| Check | Velocity L2 | Gradient L2 |
| --- | ---: | ---: |
| Initial maximum context | 4.23e-5 | 6.78e-5 |
| Initial shorter, padded conditioning | 1.31e-5 | 2.82e-5 |
| Selected NAR400 | 1.64e-5 | 2.52e-5 |

All pass the pinned numerical criteria. The listener reports **“NAR400 is
better”** on the presented fresh-score/source-score seed-831001 fixed-token
previews. This is the preferred full-data pair for this experiment. Broader
seed, dataset, vocal and natural-termination behavior remain unverified.

There are 18 acoustic audition clips: six score/seed cases, each with native
NAR, NAR100 and selected NAR400. Independent audit verifies actual prefix,
semantic IDs, seed, solver, ordered adapter stacks, finite audio, matched RMS
and peak limits. All six native re-renders are bit-identical to the preceding
AR auditions. The full page is
`outputs/koan-full-2026-10-26/nar-calibration/auditions/listening.md`.

Preferred component SHA-256 values:

- AR DoRA500: `91d0e332bb2c371ba79eed2de601dfac64dd5358db34abb69c744b677d0d55bf`.
- NAR LoRA400: `b1fd6823f176aa02b8e7ac883c5339d61c223d75774921a3a91738b9bb53385a`.

The portable local export is `models/koan-full-dora500-nar400/`: `adapter.gguf`,
`ar-only.gguf`, `prompts.json` and `metadata.json`. All 1,176 combined tensors
are byte-identical to the selected components, including DoRA magnitude/scale
and its reference fingerprint. Combined-versus-separate inference is
bit-identical on the fresh score and full external-source score. The prompt
pool contains 12 actual training style captions. Metadata records expert
families, rank, checkpoint/source hashes, trigger, required common companion,
auditioned strengths and common/base-only planning.

Refresh `http://127.0.0.1:8008/` and select
`koan-yue2-full-r32-ar-dora500-nar400`; the AR-only entry is
`koan-yue2-full-r32-ar-dora500`. This uses the existing dynamic adapter
discovery. No server restart, UI source change or default replacement was
needed. The API verifies both new entries generate, repeated fixed-seed pair
output is bit-identical, zero strength exactly matches common-only output,
scores remain fixed across the tested styles, and adapter selection appears
in result metadata. All previous adapter names and file hashes are preserved.

## Reusable starting recipe and advanced controls

The measured local starting point is 30-second real cores, per-track TXT style
captions plus `koan_palette`, tempo/key in the normalized score, rank 32 and
recording-balanced sampling. AR DoRA at LR `2e-5`, teacher KL `.2`, one neutral
replay group and an 800-update ceiling selected step 500 here; use saved
recording-level validation to select rather than treating 500 as a universal
stopping step. Plain AR LoRA remains a lower-training-VRAM alternative.

For matched acoustic refinement, freeze the selected AR and use rank-32 NAR
LoRA, LR `3e-5`, uniform noise and a 400-update ceiling with recording-balanced
heldout selection. NAR400 won both loss selection and this listening comparison.
Load the frozen common instrumental companion at strength 1 before the style;
the tested AR/NAR strengths are both 1. Keep planning common/base. Raw VAE
posterior means need no SA3-style latent normalization in this experiment.

Basic future UI controls can be dataset, adapter name, device/model and budget,
with measured context defaults and automatic checkpoint/prompt-pool publication.
Family/rank, expert learning rates and budgets, teacher KL/replay, validation
cadence/patience and separate AR/NAR inference strengths belong in advanced
controls. Record tokenizer-head, score-grid, precision, companion order and
expert-pair identities as compatibility metadata. A shared GPU reservation,
durable job registry and restart-safe cancellation still need product work.

All 194 real VAE targets and three independently authored common-model neutral
waveform targets are complete and pass `nar-target-audit.json`: exact source,
offset, frame and split joins; SHA-256 checks; shapes; finite posterior means;
context bounds; and recomputed RMS. Real target RMS ranges from 0.900 to 1.080,
with median 0.997; no latent normalization was applied. Native decode trims 64
samples from generated waveforms. Only those explicitly declared generated
targets retain the final partial posterior bin, without waveform padding. This
does not relax real-audio coverage. The first generated-target attempt rejected
the short edge and is retained in the logs; the explicit edge handling succeeds.

A bounded synthetic CPU integration check additionally prepares two exact-input
caches with the actual native builder, verifies prefix/code/clean tensors and
the selected AR companion list, reuses them without rewriting, and rejects a
different AR adapter or changed input bytes. This checks orchestration and
identity guards; it does not establish musical quality of the real dataset.

## Product boundary

The reference trainer is the local SA3 Studio implementation in
`C:/dev/sa3.cpp`, including its downstream gary4local conventions. See
[the workflow contract](yuey-training-product-contract.md). This experiment
provides versioned preparation, native fitting, run artifacts and prompt-pool
foundations. A user-facing training companion/API, durable catalog, shared
device reservation and automatic inference dropdown publication remain future
work. Neither neighboring repository is modified.

The completed AR comparison holds scores, prompts, seeds and native NAR fixed across
the old AR200 and the two new selected AR adapters, with the old AR200/NAR400
pair as an additional acoustic quality reference. No dataset audio enters those
inference auditions. The preferred pair is exported and discoverable in the
current inference UI; a user-facing training UI and durable catalog remain
future work. Neither SA3 nor gary4local is changed. No additional GGML source
change was required at the pinned 0.27 revision.
