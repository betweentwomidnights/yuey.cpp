# Expanded Koan NAR training pilot

The first expansion follows the user's
[two-seed listening feedback](nar-listening-review-2026-10-24.md). The experiment
increases training coverage while keeping the original four-track step-48
adapter as the reference for listening. Preparation, training and audition
evidence are retained under
`outputs/nar-training/koan-expanded-2026-10-24/`.

## Completed results

The 192-update run completed on the RTX 5070 Laptop 8 GB. All 12 training
tracks received optimizer updates. Sampling with replacement visited 46 of
the 48 available windows: **471.04 seconds** of distinct target cores. The two
undrawn windows are Hydroplane at 82.96 seconds and Veridian Dream at 88.96
seconds. They remain in the prepared pool; no extra updates were added to
force coverage after the declared budget.

| Update | Mean training loss | Mean six-track validation loss |
|---:|---:|---:|
| 0 | 1.17163 | 1.10917 |
| 48 | 1.13933 | 1.07796 |
| 96 | 1.12754 | 1.06853 |
| 144 | 1.12457 | 1.06696 |
| 192 | 1.12037 | 1.06645 |

Step 192 is the best of the four saved checkpoints, with a **3.85%** validation
reduction from initialization. Every validation track improves from its own
initial loss:

| Validation track | Initial | Step 192 |
|---|---:|---:|
| Aeon | 1.15431 | 1.11374 |
| Boundless | 0.996893 | 0.961707 |
| Exigence | 1.09864 | 1.04978 |
| Sentient | 1.17002 | 1.12774 |
| Silkwave | 1.13460 | 1.09126 |
| Tetsuo's Redemption | 1.10058 | 1.05450 |

The lowest *observed* validation mean was 1.06552 at update 168, which was not
one of the predeclared saved checkpoints. The slight increase by 192 and the
flattening curve are recorded; continued training is not assumed beneficial.
These percentages use the expanded six-track panel and are not directly
comparable to the earlier two-track validation mean.

The full-model oracle passed unchanged tolerances: prediction relative L2
error 0.300%, aggregate gradient error 0.630%, maximum significant individual
factor error 2.22%. All 17 application CTests and 23 Python tests passed.
Preparation took about 6.0 minutes, CPU conditioning-cache construction 13.8
minutes and GPU training 4.1 minutes on this machine. These are recorded run
times, not controlled performance comparisons. The prepared caches are retained
for further trials with this dataset configuration.

The selected delta is `run/step-192.gguf`; load it alongside the same fixed v9
companion. Its SHA-256 is
`4391b2e8b544b363eba255cb5807e5589f03066862935e075c8863f32d0be725`.
The original four-track step-48 adapter remains unchanged.

Nineteen new renders completed, including the 60-second candidate. All retain
the reference semantic streams. The 13 main A/B groups compare the original
step 48 against the selected expanded checkpoint; each includes a v9 anchor.
Exported PCM24 listening copies pass duration, finite/nonempty signal,
pairwise-RMS and peak-headroom checks. Subsequent subjective feedback is recorded
below; numerical selection alone does not assign a perceptual winner.
Local review pages are under `auditions/listening.md`, with separate
seed-831001, seed-123, long-60 and labeled checkpoint-ladder pages.

The compact portable evidence is in
[`experiments/yue2-expanded-nar-2026-10-24.json`](experiments/yue2-expanded-nar-2026-10-24.json).

## October 25 listening feedback and goal clarification

The user's seed-831001 review gives five preferences for expanded step192,
one preference for original step48 and two ties, one with an expanded lean.
The preferences for expanded on clips 05 and 07 are clear; several other
differences are small. Clip 04 was explicitly clarified as **B**, which maps
to the original step48 adapter. It is not counted as an expanded preference.

The user reports that the outputs remain musically quite similar, with changes
primarily reducing high-frequency degradation. This supports a fidelity result
and does not establish dataset-style transfer. The other four seed-123 pairs
and the long-60 comparison in this expanded set remain unreviewed.

The user clarified that the desired product is a learned dataset style that
works with new scores, unrelated source audio and standard generation, without
requiring dataset recordings as inference references. That objective and the
next evaluation gate are recorded in
[the style-transfer objective](yue2-style-transfer-objective.md). Further scaling
of fidelity-only trials is not the next milestone.

The decoded notes are in
[`experiments/yue2-expanded-feedback-2026-10-25.json`](experiments/yue2-expanded-feedback-2026-10-25.json).

## Dataset and controls

There are 18 Koan Sound tracks in the supplied `koan_dset` folder. Six remain
whole-song validation sources: Aeon, Boundless, Exigence, Sentient, Silkwave
and Tetsuo's Redemption. The other 12 train: Altercast, Cobalt, Drift,
Evocation, Hydroplane, Prism Pulse, Slyfox, Temple of Fervor, Traverse,
Traverse2, Veridian Dream and Virtue. Other artists in that folder are not
included in this run. Both keygen listening examples remain outside training.

Each training track contributes four non-overlapping 256-frame / 10.24-second
cores, placed at 10%, 35%, 60% and 85% of the valid start-position range. Every
core retains two seconds of surrounding source context during VAE encoding.
The training pool is 48 windows / **491.52 seconds**, compared with four
windows / 40.96 seconds in the first pilot. Validation keeps one window at
20 seconds from each of its six tracks.

Semantic extraction runs on each full track before cropping. Window starts
use the 25 Hz latent grid; nearby/duplicate fractions that overlap target
cores are rejected. Content hashes reject duplicate sources across the split.
The feature audit verifies non-silent source cores and excludes every source
used in the previous listening set from training. The two original validation
clips retain identical source samples and semantic tokens. Their recomputed
VAE means have approximately 4.7e-7 to 5.7e-7 relative L2 differences from the
original CPU encodings; they are not reported as bit-identical.

## Training protocol

The run starts fresh, with fixed v9 stacked after the BF16 base represented
as F16 frozen training weights. It uses rank 8 on all 196 NAR projections,
learning rate 5e-5, weight decay 0.01, global gradient clip 1, batch one and
seed 831001. AR, semantic head, VAE, norms, timestep embedding and boundary
projections remain frozen. Frozen AR conditioning is prepared on CPU as in
the original experiment. Execution uses the validated GGML 0.27 pin
`b77ae790e628270af46a8af1eca38441c23f1252`, with `GGML_CUDA_TF32=0`.

The budget is 192 stochastic updates. Checkpoints are saved at 48/96/144/192;
fixed-noise validation runs every 24 updates. History now records the sampled
window index and each validation window's loss, in command-line sample order,
alongside the mean losses. This exposes sampling coverage and track-specific
regressions. Sampling remains with replacement; the actual number of windows
visited must be reported separately from the available pool size.

Before updating parameters, a full 28-layer forward/backward probe on the new
cache shapes is checked against the independent PyTorch oracle using the
existing tolerances. Cache hashes, runtime identities, dataset identity and
exact commands are stored with the run. The original step-48 checkpoint is
retained and hash checked.

Changing both the data pool and total updates is a bounded product experiment,
not a claim that data coverage alone causes any improvement. The expanded
step-48 checkpoint is retained in the listening ladder for a step-matched
comparison.

## Audition protocol

The primary candidate is the saved checkpoint with the lowest mean fixed
validation flow loss. This is a screening metric; perceptual preference is
not assigned automatically. New renders use the previous experiment's frozen
ABC scores, captions, seeds and generation runtime. Semantic tokens must match
the preserved reference exactly.

The main blind A/B set compares the expanded candidate against the **original
four-track step-48 adapter**, with fixed v9 supplied as a separately labeled
anchor. It covers all eight seed-831001 excerpts, the four reviewed seed-123
cases and a 60-second comparison. All audio in each group is RMS matched with
a shared peak-prevention gain. A separate labeled ladder covers all four
saved checkpoints on one held-out Koan excerpt and one keygen excerpt.

This continues to test score-conditioned resynthesis. Source audio or source
VAE latents are not supplied to generation. Training still uses short windows;
neither faithful audio preservation nor natural full-song stopping is inferred
from these fixed-budget renders. The validation tracks have been used in prior
reviews and are not a new untouched test set.

## Helper support

`prepare_nar_experiment.py` accepts repeated `--window-fraction` arguments for
training windows. Holdouts continue to use `--offset`. The version-2 manifest
records explicit windows and runtime provenance; the default single-window
layout remains supported. `audition_nar_experiment.py` accepts both layouts
and records offsets in its results.

The bounded pilot still loads its conditioning caches into host memory.
Corpus-scale streaming is not implemented. The local phase driver is
`outputs/run_expanded_nar.py`; preparation/cache phases can reuse completed
artifacts when their identity matches. Fresh training has its own run identity
and does not resume the previous pilot's optimizer state.
