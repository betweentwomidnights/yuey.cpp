# Acoustic calibration for AR DoRA150

The user preferred AR DoRA over ordinary AR LoRA in the matched 150-update,
native-NAR comparison. The AR validation losses were essentially tied; the
listening preference motivates this follow-up. AR DoRA150 is now frozen while
ordinary rank-32 NAR LoRA is calibrated against it. Acoustic DoRA previously
sounded about the same as acoustic LoRA at both 100 and 400 updates, so this
experiment uses ordinary LoRA for the NAR stage.

## Scope and numerical preparation

The original 18 real 256-frame windows across 12 tracks, six whole-track
holdouts, two neutral generated replay targets, exact prefix/code/latent bytes,
rank 32, LR 3e-5, weight decay .01, clip 1, uniform noise and 400-update budget
are retained. All 392 initial factors and all 400 sampled example indices match
the historical NAR LoRA run. Neutral replay targets remain the original
AR150/native-NAR outputs; only their conditioning is recomputed. Replay is not
an independent heldout retention benchmark.

Frozen conditioning includes the common instrumental adapter followed by the
selected AR DoRA150. Old AR150 conditioning caches are explicitly rejected.
The longer BF16 cache preparation hit CUDA VMM allocation errors twice on the
8GB GPU. These logs are preserved. All 26 training/validation caches were then
prepared consistently on CPU, with unchanged input bytes. Training and listening
renders still use CUDA with TF32 disabled.

CPU preparation is **not numerically equivalent** to CUDA preparation: a matched
control has key/value relative differences of about 0.39%/0.59%, exceeding the
initial strict 1e-4 equivalence threshold. Prefix, codes and clean target
latents are byte-identical. The failed strict check is retained explicitly.
Therefore this is a practical calibration using CPU conditioning, not a pure
one-variable numerical comparison with the historical CUDA-cache run. Its own
initial and selected full-model dense CPU forward/gradient oracles pass; the
CUDA renderer auditions are the relevant transfer check.

## Selection and listening

| Updates | Heldout real flow MSE |
| ---: | ---: |
| 0 | 1.36807 |
| 100 | 1.10095 |
| 200 | 1.08505 |
| 300 | 1.08048 |
| 400 | 1.07936 |

The lowest saved heldout flow MSE selects **step 400**,
from 1.36807 initially to
1.07936. Loss measures acoustic target fitting,
not preservation of melody or quality of Koan sound design. At the early
100-update audition, the user rated calibration **“About the same”** as native
NAR. On the final source comparison, the user reports **“calibrated NAR400 is
slightly better fidelity.”** This supports AR DoRA150 plus ordinary NAR LoRA400
as the preferred candidate from the current listening comparisons. The response
does not separately establish better melody/chord preservation or broad-seed
superiority. The earlier positive AR rating remains a separate result.

The primary auditions hold AR DoRA150, saved semantic token IDs, exact prefix,
flow seed, duration and 32-step solver fixed. Only NAR adaptation changes.
Native-NAR replay is bit-identical to the earlier AR DoRA auditions. Four cases
cover a fresh score, the external piano/orchestral source at two seeds, and
standard generation. No reference latents enter inference. Previews are RMS
matched per case. The older AR LoRA150/NAR400 is included as a separate reference
with different codes and acoustic weights, not as a controlled primary arm.

Local listening set: `outputs/koan-ar-dora-nar-calibration-2026-10-26/auditions/listening.md`.
Selected NAR SHA-256: `11ff98dfbcd17e1a53b913b99d4cbff5b765eabf08384917f216b6babf040b63`.

No new native code or GGML edits were needed. The previously passing 19 native
application and 23 Python tests are recorded as prior checks, not claimed as
new runs. Existing web presets remain unchanged; the live server has not been
republished with the AR DoRA loader. This pair is not yet a UI default.
The next milestone is to expose this validated candidate for manual web UI
experiments before further training. Local pair identity and constraints are
recorded in `outputs/koan-ar-dora-nar-calibration-2026-10-26/preferred-pair.json`.

The subsequent [web UI rollout](koan-ar-dora-webui-2026-10-26.md) publishes the
AR-only and calibrated pairs on port 8008 alongside the older LoRA presets.
The statement above describes the calibration run before that rollout.

[AR DoRA experiment](koan-ar-dora-2026-10-26.md) and
[portable evidence](experiments/yue2-koan-ar-dora-nar-calibration-2026-10-26.json).
Commands, numerical failures, diagnostics and source snapshots remain in the
experiment output directory.
