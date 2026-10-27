# sv-billie YuE2 adapter review

The useful finding is that this released artist adapter modifies both AR and
NAR transformer experts. It is relevant to instrumental adaptation as well as
the user's reported vocalist resemblance: these are general music-generation
projections, not a voice-specific head. It supports investigating a restrained
AR adaptation with matched acoustic adaptation, rather than assuming that
more AR updates alone should deliver both style and fidelity.

The user's clarification changes our acceptance criterion: descriptive Koan
style prompts and learned trigger words are valid production conditioning.
Neutral captions remain a diagnostic. Dataset reference recordings must still
be unnecessary at inference, and supplied musical content must remain useful.
The [policy record](experiments/yue2-prompted-style-policy-2026-10-25.json)
preserves that instruction. This review does not start another training run.

## What was inspected

Model repository: [HaileyStorm/sv-billie-yue2-lora](https://huggingface.co/HaileyStorm/sv-billie-yue2-lora).
Pinned revision: `94fe9fbf5dfd3e97eefeb3ef54e2801eb3b69077`.
The full 117,500,888-byte safetensors file was downloaded as data, its SHA-256
verified against the Hub LFS object, and its factors inspected without importing
or executing model-repository Python. The metadata, header, example manifest
and style sidecars are retained in `outputs/sv-billie-review-2026-10-25/`.
Published audio was not downloaded for a listening benchmark.

Checkpoint file SHA-256:
`e461611cd0e9d424b1dcb163b7460ba133681e6d55d3d0156a59039615c6235f`.

| Property | Published Billie adapter | Current Koan pilot dataset delta |
|---|---|---|
| Expert scope | AR and NAR, all 28 layers each | AR only; NAR frozen |
| Rank | 32 in both experts | 8 |
| Target layout | Fused QKV, attention output, fused gate/up, MLP down | Separate Q/K/V/O and gate/up/down |
| Saved factors | 448 BF16 tensors | 392 F32 tensors |
| Learned factor parameters | 58,720,256 | 8,716,288 |
| Embedding, LM head, norms, VAE/boundary tensors in adapter | None | None |
| Selected checkpoint | Metadata reports step 950, epoch 19 | Step 32; step 128 is an overfit diagnostic |

The published file has 112 fused target pairs per expert, 29,360,128 factor
parameters per expert. Every A and B factor is finite and nonzero. The two
experts' export aliases are `text_encoders.*` for AR and `diffusion_model.*`
for NAR, as confirmed by the [toolkit key mapping](https://github.com/ostris/ai-toolkit/blob/main/extensions_built_in/audio_models/yue2/yue2_model.py).
These are tensor observations, not an inference from the card's loader advice.

The parameter counts describe the newly trained dataset delta. Our common
instrumental AR companion is a separate fixed adapter, so these are not equal
base stacks or a controlled capacity/quality comparison. Fused projections
share A factors, so rank 32 here is not simply four times our independent
rank-8 parameterization. Converting to our separate projection layout would
duplicate shared A factors without changing the represented function.

No audio boundary replacements or other non-LoRA tensors are included. This
does not establish every unpublished training setting or rule out a merged
companion during training. The published use instructions describe loading
the artist adapter with the base model and VAE.

## What the learned matrix ranks suggest

I computed singular spectra of each effective update B@A using QR factors and
a small core SVD, without materializing every dense update. I also split the
QKV and gate/up output rows to match our native independent projections before
comparing rank concentration. The medians over 196 projections per expert are:

| Expert | Update energy retained by leading 8 directions | Rank for 95% energy |
|---|---:|---:|
| AR | 96.79% | 5 |
| NAR | 88.61% | 16 |

This is a reason to test asymmetric capacity, such as retaining modest AR rank
while trying rank 16 or 32 for acoustic calibration. It does not prove that
rank 8 can learn every vocalist/style, that NAR alone creates the perceived
resemblance, or that discarding a small-energy direction is perceptually safe.
These are medians of weight-space approximations, not audio-quality scores.
The raw unscaled update norms likewise cannot rank adapter quality. Exact
training alpha is not published in the inspected metadata.

## Examples and inference settings

The [model card](https://huggingface.co/HaileyStorm/sv-billie-yue2-lora) describes
curated examples selected across three seeds and three sampler configurations,
including requests for which every take was discarded. It requests a trigger
and detailed style conditioning. This is valid use under our clarified goal;
matched adapter-off prompts are still needed to measure the LoRA's contribution.

The [album manifest](https://huggingface.co/HaileyStorm/sv-billie-yue2-lora/blob/main/examples/the-long-way-back/album.json)
shows 100 ODE steps for the seven step-950 tracks, with semantic temperatures
between 0.80 and 0.94. One other track uses step 900; the final track adds a
DreamPop adapter at strength 0.3. The first recommended examples use the released
step-950 adapter alone. No aggregate success rate or matched base-model/ACE-Step
benchmark is supplied. Curation is explicitly disclosed, not evidence that the
examples should be dismissed.

The style sidecars specify more than an identity tag: vocal delivery, register,
instrumentation, rhythmic arrangement and section-level changes. Their presence
does not tell us how much the base model could already produce. The user's
positive resemblance judgment is useful feedback; this review does not claim
an independent listening comparison or rank YuE2 above their ACE-Step or SA3
results. An instrumental timbre/production transfer with content preservation
is also a different evaluation task from recognizable vocal character.

## Training mechanisms worth borrowing, with limits

The current [AI Toolkit implementation](https://github.com/ostris/ai-toolkit/blob/main/extensions_built_in/audio_models/yue2/yue2_model.py)
provides separate AR cross-entropy and NAR flow losses. When AR CE is enabled,
the acoustic cache is detached: NAR flow loss does not backpropagate through
the AR expert. This makes a staged native approach plausible without requiring
end-to-end gradients through both transformers simultaneously.

That implementation supports KL(base || adapted AR) and an AR learning-rate
multiplier. Its [UI](https://github.com/ostris/ai-toolkit/blob/main/extensions_built_in/audio_models/ui.tsx)
warns about fast AR memorization; its UI default KL weight is 0.2, while the
model-class fallback is 0.0. Those are current software settings, **not proof
that the Billie run used them**. The checkpoint reports toolkit version 0.13.15
but no training commit/config; a matching `v0.13.15` Git ref was not available
in this review. Source blob SHAs and read-only excerpts are archived.

The toolkit also distinguishes full-song AR supervision from random NAR
windows, supports separate stem losses, and exposes configurable score dropout.
Actual use of those options, corpus size/duration, optimizer, learning rates,
window length, exact semantic head variant and companion merging are not
established by the public release. It points to the same MotherSuperior tokenizer
repository we investigated; this is not evidence of a newly released official
or lossless semantic tokenizer. Do not infer a song count from step 950/epoch 19
or copy 950 updates as a training prescription.

## Consequences for the next Koan experiment

1. Teach a consistent trigger alongside useful style descriptions. A proposed
   alias is `koan_palette`; no learned association with that new alias has been
   established in our existing checkpoints. It can use the existing tokenizer;
   the published adapter also contains no new embedding table.
2. Evaluate identical prompted scores with the adapter on/off. Add no-trigger
   and neutral controls as diagnostics, not requirements to accept a useful
   prompted workflow. Do not supply a dataset recording as a style reference.
3. Keep the early AR reference and improve content retention using broader
   score-supervised data, early stopping and a separately verified retention
   objective. Larger AR rank and more updates are not the first remedy for
   the observed melody loss.
4. Give NAR a matched calibration experiment with the selected AR fixed, exact
   deployment prefixes/companions, and clean real-audio VAE targets. Test larger
   NAR rank as a controlled capacity variable while keeping the VAE frozen.
   This is motivated by the joint artifact, not a guaranteed fidelity repair.
5. Test 32 versus 100 midpoint steps on fixed prefix/tokens/noise using the same
   renderer. The current [toolkit pipeline](https://github.com/ostris/ai-toolkit/blob/main/extensions_built_in/audio_models/yue2/src/pipeline.py)
   uses midpoint integration too. That isolates numerical rendering from AR
   content decisions; more ODE steps cannot repair already incorrect AR tokens.

Our converter currently supports separate native/PEFT projection names, not
this fused Comfy layout. A faithful import requires mapping both expert aliases,
splitting QKV B rows as 2048/1024/1024 and gate/up rows as 6144/6144, preserving
the shared A factors and checking scaling conventions. No GGML kernel change
is implied. An eventual AR-only/NAR-only/both ablation should use identical
prompts and seeds. Do not stack our no-vocals instrumental companion when
assessing this vocalist adapter.

Portable inspection and proposal:
[`experiments/yue2-sv-billie-review-2026-10-25.json`](experiments/yue2-sv-billie-review-2026-10-25.json).
No conversion, synthesis, new LoRA training or ACE-Step benchmark was performed
as part of this inspection.

A later [AI Toolkit source review](ai-toolkit-yue2-review-2026-10-25.md) compares
the completed rank-32 Koan experiment with the pinned upstream implementation,
including tokenizer head identity, exact AR loss targets, context coverage and
remaining acoustic/DoRA experiments. This document retains the earlier inspection.
