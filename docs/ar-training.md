# Experimental semantic AR LoRA and DoRA training

The [full-corpus Koan experiment](koan-full-training-2026-10-26.md) completed
matched rank-32 LoRA/DoRA on 30-second scored windows. It documents
42-recording preparation, real TXT captions, score conversion checks, measured
30/60-second resource behavior and reusable manifest/grouped-sampling support.
The listener prefers full-data DoRA500 with its matched NAR LoRA400 calibration.

This is a native, batch-one trainer for YuE2's semantic AR stage, with fixed
target-frame counts and masked variable-length prefixes.
It follows the [failed NAR-only style gate](nar-style-gate-2026-10-25.md).
Its generated-label controls test whether the training path works and can
change fresh semantic generations. **They are not Koan-trained adapters.**
The real-audio supervision audit remains a separate listening gate.

## Scope

`yue2-ar-train` adds configurable-rank F32 LoRA factors to all seven standard AR projections
in all 28 layers: Q/K/V/O and gate/up/down. At rank 8 this is 392 factors and
8,716,288 parameters. It keeps embeddings, norms, LM head, NAR, audio boundary
projections, semantic tokenizer and VAE frozen. The fixed instrumental AR
adapter is included during both training and audition inference.

The objective is teacher-forced cross-entropy over MUSIC_END and the 32,768
codec tokens, matching the restricted semantic sampling vocabulary. Prefix
tokens supply causal context but contribute no loss. Code targets are shifted
by one input position, and the loss averages over target positions only.
The first target is predicted from the prefix's final MUSIC_START token.

`--predict-end` adds a final EOS target only for genuinely complete sequences.
It is off for cropped examples, including these 256-frame capacity probes.
Synthetic EOS fixtures separately validate the end-token case.

Frozen unquantized weights use F16 storage, with BF16 rounded through F16.
AR training casts projection weights within each checkpointed graph to F32
and requests F32 matrix multiplication precision. Norms and trainable factors
also compute in F32. Inference keeps its existing arithmetic. This avoids
retaining the whole AR model in F32 on the GPU. Host AdamW uses beta values
0.9/0.999, epsilon 1e-8, global gradient clipping, and decoupled weight decay.

`--dora` adds an output magnitude per projection row and differentiates through
the complete direction norm. The frozen reference includes the instrumental
companion: both its output and its contribution to the row norm are included.
Fixed adapters must be ordinary additive LoRA; nested DoRA and replacement
weights are rejected. The KL teacher remains common-only, including after
resuming a nonzero-magnitude checkpoint. LoRA and DoRA states cannot be mixed.

DoRA exports store raw magnitudes and inference row gains. A compatibility
fingerprint binds the required ordered frozen adapter prefix and its strengths;
load that companion before the learned DoRA at the same strength used for
training. The fingerprint is a compatibility check, not a cryptographic model
identity; experiment manifests also record SHA-256 hashes. Inference strength
blends the whole normalized delta. See the
[matched AR comparison](koan-ar-dora-2026-10-26.md) and the separate
[NAR comparison](koan-nar-dora-2026-10-26.md).

Per-block recomputation saves 29 hidden boundaries and shares graph workspaces
between layers of the same shape. Manual causal attention keeps the backward
path functional, without inference's in-place KV-cache writes. The full
256-frame, 223-prefix-token F32-compute update was observed at **5,396 MiB**
device memory on the 8 GB RTX 5070 Laptop GPU. Sampling the device memory every
half-second is an observation, not a guarantee for larger examples.

No new GGML changes were needed for these probes at submodule revision
`b77ae790e628270af46a8af1eca38441c23f1252` (0.27).

## Correctness gates and precision finding

`tools/check_ar_training_reference.py` independently implements full-prefix
causal GQA, RoPE, frozen companion factors, target-only loss and all adapter
gradients in CPU/PyTorch, reading only GGUF tensors. It also compares a first
AdamW update against PyTorch using the native gradients.

The first F16-projection prototype passed its initial oracle, but its gradient
comparison after near-converged fitting failed the unchanged 5% tolerance:
aggregate relative error was 7.76%, maximum significant-factor error 23.7%.
Losses and logits were close. Those runs remain evidence of the precision
limit and are not the checkpoints selected for auditions.

With F32 projection compute, the initial full-model oracle reports 0.000369%
relative logit error, 0.00285% aggregate gradient error, 0.0104% maximum
significant-factor gradient error, and a 7.45e-9 maximum AdamW factor difference.
The experiment also checks gradients with both factors nonzero after fitting,
CPU/CUDA EOS behavior, exact two-step resume, and a zero-adapter inference
control. The [portable evidence](experiments/yue2-ar-semantic-probe-2026-10-25.json)
contains the final results rather than treating a low training loss as proof
of correctness or perceptual style.

The final F32-compute run passes after fitting too: aggregate gradient error
is 0.0156%, maximum significant-factor error 0.0702%, and relative logit error
0.000771%. CPU and CUDA EOS fixtures pass, and uninterrupted/resumed two-step
state files are byte-identical. The zero adapter reproduces both sampled
tokens and rendered WAV bytes exactly. After rebuilding the application tests,
17 application CTests and 23 Python tests pass. The separate full GGML CTest
suite was not built and is not included in that count.

## Generated-label experiments

The primary capacity example uses 256 known generated frames from the fresh
waltz score in the earlier style gate, with a neutral caption and no EOS on
the crop. It trains for 64 updates at learning rate 1e-4, weight decay .01,
clip 1 and rank 8. It does not use the audio tokenizer or dataset recordings.
This is deliberate memorization, not a validation split or a quality recipe.

A second synthetic control generates an electronic target with the same score
and an explicit glitch-hop/FM-bass/breakbeat caption. A fresh optimizer and
adapter then train on those known codes under the **neutral** caption. Fresh
auditions use neutral text, a new seed and no reference codes/audio. A second
score is excluded from both training examples. Explicitly labeled teacher
and fixed-target-code renders expose the target and decoder ceiling; they
are not counted as fresh-generation results.

| Control | Initial cross-entropy | After 64 updates |
|---|---:|---:|
| Neutral generated target | 3.49975 | 0.0286995 |
| Electronic generated target, neutral training caption | 4.26281 | 0.0326234 |

At fresh seed 123 the neutral memorization control reproduces 99.6% of its
training target IDs; the common model matches 0.39%. This is strong evidence
of memorization capacity, not musical generalization. The electronic fit changes
all sampled token IDs relative to the common model in the familiar- and
unseen-score auditions. Token differences do not decide whether the resulting
sound palette is useful, whether the score survives, or whether it matches the
dataset. The user subsequently confirmed that the sound palette changed in the
presented unseen-score pair. That confirms an audible change on a new score;
direction toward the teacher palette, score adherence, fidelity and Koan resemblance were not assigned by
that feedback. See the [recorded feedback](experiments/yue2-ar-palette-feedback-2026-10-25.json).

Judge audible change, sound-palette movement, score adherence and artifacts
separately. Predicting a stored motif, token differences or a lower loss does
not prove style transfer. No Koan resemblance, ordinary free-planning quality
or dataset generalization is assigned without listening.

Local auditions: `outputs/ar-semantic-f32-2026-10-25/auditions/listening.md`.
Commands, model/runtime identities, raw WAVs, tokens, factors, optimizer state,
diagnostics and independent oracle results remain alongside the experiment.

## Semantic-target audit

`outputs/ar-target-audit-2026-10-25/listening.md` compares two Koan and two
keygen excerpts with VAE-only, native-NAR and fixed-v9 reconstructions. Both
semantic-decoder arms use full-song-context audio-derived tokens, the same
transcribed instrumental score, neutral caption, fixed instrumental AR adapter
and flow seed. Existing full-context tokens and posterior-mean VAE controls
are inherited from the earlier audit with their provenance retained.

A generated-code/audio control also decodes known tokens, re-encodes the audio
with the MotherSuperior head, and renders those inferred tokens through native
NAR and v9. It recovers 11% of the original token IDs. Non-unique codes and the
decoder mapping limit what that statistic means: it is not a perceptual style
score or proof that labels are unusable. It does show the head is not an exact
token-sequence inverse on this example. Good VAE-only reconstruction does not
settle semantic-label quality. The [portable target audit](experiments/yue2-ar-target-audit-2026-10-25.json)
records numerical controls separately. The user reports that these
reconstructions retain rhythmic quality but raise degradation concerns, and
explicitly chose to try a small pilot anyway. This authorizes an exploratory
noisy-label test; it is not a target-quality pass. See the
[target decision](experiments/yue2-ar-target-feedback-2026-10-25.json) and
[two-track Koan pilot](koan-ar-pilot-2026-10-25.md).

## Reproduction

Use an already transformed **used score** exported by generation. The prefix
helper adds the same instrumental caption boilerplate and lyric section tags;
it does not transcribe audio or transform a raw score's vocal lane.

```powershell
$env:GGML_CUDA_TF32 = '0'
build-ggml027-cuda/bin/Release/yue2-ar-train --prefix-out NEW/prefix.i32 `
  --tokenizer models/YuE2-3B-GGUF/sidecars/yue2-qwen.tiktoken `
  --abc used.abc --prompt 'Instrumental music.' --instrumental

build-ggml027-cuda/bin/Release/yue2-ar-train `
  --model models/YuE2-3B-GGUF/yue2-3.6B-v1.0-BF16.gguf `
  --prefix NEW/prefix.i32 --tokens known-codes.i32 `
  --fixed-lora models/yue2-instrumental-cot-full-v1.0-F16-LoRA.gguf `
  --out NEW/run --identity HASH_OF_MODEL_DATA_COMPANION_AND_RUNTIME `
  --device cuda --rank 8 --lr .0001 --weight-decay .01 --clip 1 `
  --steps 64 --save-every 32 --diagnostics NEW/initial-diagnostics.gguf
```

The identity must cover exact prefix and code bytes, model/companion hashes,
target/end policy and operative runtime/precision. Resume requires matching
configuration and identity, uses `--resume step-N.state.gguf`, and writes into
a fresh directory. Steps mean additional updates. State contains factors,
moments, optimizer step, complete sampler state and early-stopping history.

`yue2-ar-sample` samples a bounded fresh semantic stream directly from an exact
prefix and renders it. It does no planning and accepts no reference audio or
semantic continuation. Repeat `--lora` for the common instrumental adapter and
the new AR delta. `yue2-render-tokens --prefix prefix.i32` separately renders
known codes with the exact conditioning; `--prefix` is mutually exclusive
with its text/ABC/lyrics options.

`yue2-ar-sample --temperature T` optionally changes semantic sampling (default
1.0, positive and finite). Fixed-token midpoint controls must instead use
`yue2-render-tokens` so semantic samples do not change with decoder settings.

## Limits and next decision

The trainer now accepts repeated `--sample-prefix` / `--sample-tokens` pairs
of the same target-frame count. Shorter prefixes are left padded, masked out of
real-token attention and assigned real positions starting at zero. Padding
queries keep finite causal attention; they cannot influence scored rows.
`--validation-prefix` / `--validation-tokens` and `--retention-prefix` /
`--retention-tokens` supply evaluation-only controls. The CLI logs per-example
losses, sample indices and means. A uniform sample scheduler saves mt19937 state
for exact resume. `--shuffle-samples` instead visits each prepared view once
per shuffled epoch, saving its permutation and cursor. `--max-prefix` can
reserve extra prefix slots explicitly.

Score dropout uses separate prepared prefix views, removing the ABC interior
while retaining the full-mode instruction, lyric tags and boundary markers.
It does not switch to the unrelated symbolic-off conditioning recipe. The
four-view two-track pilot samples scored/dropout variants with equal
probability. Checkpoints are saved at evaluation intervals; the experiment
driver selects among saved checkpoints by held-out loss.

`--kl-weight W` adds `W * KL(common teacher || adapted student)` on training
rows. The teacher includes fixed companions and is cached before any learned
update or checkpoint load. The full restricted-vocabulary probability targets
are immutable and normalized. By default they are held in RAM and regenerated
on resume. `--teacher-cache DIR` instead streams one
`DIR/teacher-NNNNN.gguf` window at a time; `--prepare-teacher-only` writes the
missing files atomically and exits. Each file records the coordinator teacher
identity, base/companion SHA-256s, device, frames, AR slots and the window's
exact prefix/codes (plus manifest SHA-256s). Mismatches are rejected. See
[the teacher cache measurements](ar-teacher-cache-2026-10-28.md). Validation/retention
losses remain plain semantic cross-entropy; retention inputs remain evaluation
only. Replay supplies ordinary training examples explicitly, with their own
prefixes and token labels. The independent oracle also recomputes the teacher
distribution and checks nonzero-KL gradients.

`--early-stop-patience N --early-stop-min-steps M` stops after N scheduled
held-out evaluations without improvement once M updates have elapsed. Stopping
history is checkpointed; the driver still selects a saved checkpoint by
held-out loss and listening remains a separate decision.

This remains an in-memory prototype. Variable frame counts, batching, corpus
streaming and automatic perceptual model selection are not implemented.
These controlled fits are not a replacement for the user's
dataset-style objective or proof that degraded semantic targets are adequate.

AR weights are shared with planning even though there is no planning loss.
Use a base/common planning pass and supply its score to the adapted semantic
pass. The exact-prefix auditions enforce that separation. Fixed decoder weights
also do not imply fixed decoder conditioning: adapted AR states feed the NAR
decoder as well as changing sampled codes. Attributing an effect solely to
code IDs requires additional fixed-code controls.
