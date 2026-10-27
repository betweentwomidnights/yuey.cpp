# AR DoRA controlled experiment

The user correctly pointed out that the earlier acoustic DoRA comparison kept
an ordinary LoRA AR model fixed. That result did not test DoRA's ability to
learn semantic/style behavior. This experiment changes the AR adapter family
while keeping the original AR150 training recipe and native NAR fixed.

## Recipe and result

Rank 32, all seven projections in all 28 AR layers, LR 2e-5, KL weight .2,
weight decay .01, global clip 1 and shuffled epochs. Training uses the original
18 unique 256-frame excerpts across 12 tracks (184.32 s), full and hybrid
score-dropout views, and two neutral replay examples. Six whole-track heldouts
have both prefix views; neutral retention is separate and evaluation-only.
The caption retains the `koan_palette` trigger. This is the old AR150 recipe,
not the newer expanded scored-only AR200 recipe. The tokenizer labels and
restricted semantic objective are unchanged; no direct ABC/planning loss or
mid-crop EOS supervision is added.

All 392 initial A/B factors and all 400 sampled example indices match the
ordinary LoRA control. DoRA additionally learns 573,440 output magnitudes.
The maximum budget is 400 updates, with validation/checkpoints every 50 and
the same early-stopping policy. Both methods select step 150 by heldout CE.

| Updates | LoRA heldout CE | DoRA heldout CE |
| ---: | ---: | ---: |
| 0 | 6.369936 | 6.369936 |
| 50 | 6.310524 | 6.309701 |
| 100 | 6.283658 | 6.283305 |
| 150 | 6.276298 | 6.276387 |
| 200 | 6.297864 | 6.298710 |
| 250 | 6.368977 | 6.370980 |
| 300 | 6.466778 | 6.469979 |
| 350 | 6.624750 | 6.629637 |
| 400 | 6.826228 | 6.831969 |

At selection, LoRA is 6.276297530 and DoRA 6.276386778: effectively tied.
Neutral retention is likewise close, 3.744019270 versus 3.744098425.
Both overfit after step 150. No perceptual superiority is inferred from these
losses. The user subsequently rated the matched AR150 comparison
**“AR DoRA is better”**, supporting AR DoRA150 as the candidate for matched
acoustic calibration. This is a listening preference on this comparison, not a
broad-seed result. The user's earlier
“about the same” ratings concerned NAR DoRA only.

## Companion-aware DoRA

The direction is the effective frozen weight (base plus instrumental LoRA)
plus the learned B@A. Its complete row norm participates in backpropagation,
as in our SA3 implementation; this differs from a detached-norm convention.
The low-rank norm identity includes the common companion without retaining a
dense merged copy of every adapted model weight. The KL teacher uses a separate
common-only forward path, excluding learned magnitudes even after resume.

Exports contain A/B, raw magnitudes and precomputed inference gains. A path-free
FNV1a64-v1 compatibility fingerprint requires the same ordered frozen adapter
prefix and strengths before the learned DoRA. This is not a cryptographic
identity; SHA-256 model and data hashes are separately recorded. Missing,
changed-strength or reordered references are rejected, as are incompatible
overlaps/nested DoRA. Fixed replacement adapters are not supported for this
training mode. Strength blends the entire normalized delta, with exact base
behavior at zero. LoRA and DoRA optimizer states are distinct.

## Validation and auditions

Independent dense CPU/PyTorch references include the frozen companion, teacher,
full norm gradients and first AdamW update. Full-model gradient relative errors
are 1.98e-5 initially and 2.42e-5 at the selected checkpoint. DoRA resume is
byte-identical; the ordinary LoRA fixture state remains byte-identical to its
historical result. All 196 exported row gains match an independent dense
reference, maximum absolute error 2.38e-7. All 19 native application tests and
23 Python tests pass, including CPU/CUDA loader and normalization coverage.
No GGML changes were needed.

The primary pair uses LoRA150 versus DoRA150 with the same exact score/prompt
prefix, common companion, semantic seed, forced duration and native NAR weights.
Each AR samples its own codes and supplies decoder conditioning, so semantic
tokens are allowed to differ. Autoregressive divergence under a common seed
does not by itself establish better style or content preservation.

Four cases cover a fresh authored score, the user's external piano/orchestral
source at seeds 831001 and 123, and standard generation. No reference audio
latents or Koan recording enters these renders. Source remix remains a
transcription-to-score-to-resynthesis workflow. The recording is outside the
training files; underlying composition provenance is not established. Previews
are RMS matched per case with a common peak-safe gain. A third arm, the prior
AR150/NAR400, is a separate quality reference with different acoustic weights.
The legacy fresh-score LoRA tokens and WAV reproduce bit-for-bit with the new
loader. This small audition set is not a broad-seed evaluation.

Local listening page: `outputs/koan-ar-dora-2026-10-26/auditions/listening.md`.
Selected adapter: `outputs/koan-ar-dora-2026-10-26/run-0100/step-150.gguf`,
SHA-256 `1f97762c1d8656f653b7fd0cd3642f5f57bec82c877e4e19b4b38b4ffa3758ca`.
The earlier fresh-score audition is the same selected checkpoint.

The current live web server and existing LoRA presets are unchanged. DoRA has
not been published to the UI; that running server predates this loader and
would need rebuilding/restarting before a DoRA preset is offered. Do not treat
the older AR150 NAR400 as calibrated for the new AR DoRA. A matched acoustic
calibration is a subsequent experiment if listening supports it.

The user's positive AR assessment motivated the subsequent
[matched acoustic calibration](koan-ar-dora-nar-calibration-2026-10-26.md).
Its 100-update acoustic audition was rated **“About the same”** as native NAR.
That follow-up uses consistent CPU cache preparation after CUDA allocation
failures; its numerical preparation limitation is documented separately.

[Portable evidence](experiments/yue2-koan-ar-dora-2026-10-26.json),
[NAR DoRA comparison](koan-nar-dora-2026-10-26.md),
[trainer details](ar-training.md), and
[future control boundaries](yue2-adapter-controls.md).
Commands, logs, diagnostics and immutable source snapshots remain under the
experiment output directory.
