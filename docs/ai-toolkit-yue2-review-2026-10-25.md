# AI Toolkit YuE2 training review

The native trainer has the main building blocks. Our bounded Koan pilot is still
a materially different training recipe from AI Toolkit. The most useful next
work is label validation and better data/context coverage, followed by controlled
acoustic experiments. Increasing rank or copying 950 updates is insufficient.

This is a read-only review of `ostris/ai-toolkit` at
[`73ebae13a468942ded8b47991b5d56eeeea25c36`](https://github.com/ostris/ai-toolkit/tree/73ebae13a468942ded8b47991b5d56eeeea25c36),
whose tip commit is dated October 24. The YuE2 model, backbone and pipeline blobs
match the files inspected in our earlier Billie review. The second pass traces
the precise loss targets and the surrounding dataset, network and scheduler
implementation, and checks relevant upstream discussions. Current source is not
proof of the Billie run's unpublished configuration or exact training commit.
Downloaded Python was archived as text, never imported or executed. No training
was launched, model changed, or running web UI restarted for this review.

The user's current assessment is that YuE2 has made useful progress, while their
SA3 DoRA adapters capture this dataset's sound design better. That is listening
feedback, not a controlled attribution of the difference to DoRA alone.

## Material differences

| Area | AI Toolkit inspected path | Our rank-32 Koan experiment |
| --- | --- | --- |
| Real-audio semantic labels | v4 head by default; later heads configurable | v9 BF16 head, verified by source SHA |
| AR context | Song from its beginning; optional length cap | Fixed 256-frame excerpts, including mid-song excerpts |
| AR prediction targets | ABC, ABC_END, MUSIC_START, codec sequence; MUSIC_END when complete | Codec sequence only; cropped Koan examples have no positive EOS target |
| AR vocabulary in training | Full LM-head vocabulary | MUSIC_END plus 32,768 codec classes |
| NAR context | Random window from whole-song cache; default maximum 1,500 frames | Same 18 fixed real windows of 256 frames |
| Score dropout | Rebuilds a real symbolic-off instruction/prefix | Removes score interior but retains the full-mode instruction |
| Acoustic noise times | YuE2 UI defaults to sigmoid-distributed times | Uniform sigma in [.05,.95] |
| Expert optimization | Both updated in one run; NAR cache detached from AR gradients | AR selected first, then fixed during NAR calibration |
| Captions | Per-item captioning tools and native style/lyrics layout | One shared Koan caption plus trigger for all real windows |
| Base stack | Native NAR by default; optional community NAR merge | Native NAR with fixed instrumental AR companion |

The source for AR inputs/loss routing is
[`yue2_model.py`](https://github.com/ostris/ai-toolkit/blob/73ebae13a468942ded8b47991b5d56eeeea25c36/extensions_built_in/audio_models/yue2/yue2_model.py#L506-L705).
The broader training intentions and defaults are also described in the
[YuE2 UI configuration](https://github.com/ostris/ai-toolkit/blob/73ebae13a468942ded8b47991b5d56eeeea25c36/extensions_built_in/audio_models/ui.tsx#L73-L221).
These are differences to test, not a list of settings proven superior.

## 1. Validate the head choice before another substantial fit

The local GGUF records head SHA-256
`cffb673cd663517c8978ba73d186a87517958025f7e409371ea56df3bb364db2`.
The model repository's LFS listing identifies that exact object as
`tokenizer_head_joint_v9.bf16.safetensors`. AI Toolkit's
[default head](https://github.com/ostris/ai-toolkit/blob/73ebae13a468942ded8b47991b5d56eeeea25c36/extensions_built_in/audio_models/yue2/src/tokenizer.py#L23-L30)
is `tokenizer_head_joint_v4.pt`. The repository name ending in `v4` does not
identify which head file a trainer actually used.

[Merged PR 1055](https://github.com/ostris/ai-toolkit/pull/1055) deliberately retained
v4 as the default while adding later-head support and separate cache keys. Its
author reported only 22–31% frame agreement between v4 and v9 on three songs;
that is their experiment, not a result on our Koan data or a quality score.
A follow-up confirms a v9 training run completed, without supplying a decisive
listening comparison. Billie metadata does not establish its head variant.

Our latest fit uses v9 labels with native NAR as the starting decoder, without
the v9 companion. That was deliberate after the earlier target auditions, but
it does not reproduce the toolkit default. A head co-trained with an adapted
decoder need not be the best label source for a different starting decoder.
That is a hypothesis to check, not a diagnosis of the present adapter.

The first next experiment should therefore be a small target audit: identical
Koan passages, scores, prompts, flow noise and renderer; compare v4/native-NAR
against v9/native-NAR, with VAE-only and explicitly matched head/companion
controls. Rate sound-design retention, musical information and degradation
separately. Do this before training. Keep head-specific caches and hashes;
do not relabel or overwrite the present v9 experiment. A training-label audit
does not require changing the UI's configured real-audio continuation tokenizer.

There is a smaller preprocessing parity difference too:
[toolkit normalization](https://github.com/ostris/ai-toolkit/blob/73ebae13a468942ded8b47991b5d56eeeea25c36/extensions_built_in/audio_models/yue2/src/tokenizer.py#L221-L245)
uses PyTorch's default sample standard deviation, while
`src/mert2_encoder.cpp` divides variance by T. Ignoring epsilon, their denominator
ratio is sqrt(T/(T-1)): about 0.011% at T=4,500. Since we tokenize full tracks,
this is usually small. It merits a fixed-feature token/logit parity check,
not a claim that it explains the audible reconstruction problem. MERT precision,
resampling and head version must be held fixed during that check.

## 2. Match coverage and musical context, not the step number

Our real training pool is 184.32 seconds: 18 excerpts across 12 training tracks,
with six separate tracks held out. Repeatedly presenting the same excerpts with
new Gaussian noise does not expose additional arrangements or sound-design events.
Toolkit caches songs and draws new NAR spans; a default 1,500 frames means up to
60 seconds rather than our 10.24 seconds. Its AR view still begins at the song
start even when the acoustic span comes from later in the recording.

For our instrumental score-following objective, increase distinct audio coverage
and use accurate local scores. Profile 512/750-frame acoustic windows before
trying the toolkit's full default on an 8 GB GPU. Longer is an experiment, not
a guarantee: it costs attention memory and can expose more transcription errors.
Keep the current whole-track holdouts and evaluate multiple unrelated input scores.

AR should also see complete musical units and genuine end targets. Native
`--predict-end` already exists; it was correctly disabled for arbitrary crops.
Do not append EOS to every middle-of-song excerpt just to make sampling stop.
Use genuinely complete songs, or intentionally constructed complete phrases
whose score and audio ending agree. The UI overrun observation does not prove
that absent EOS supervision caused the problem; generated-score safety margins
already exist in the base pipeline. Measure termination on matched base/adapted
requests rather than declaring the toolkit has solved it.

## 3. The AR objectives are not numerically equivalent

Toolkit `_ar_inputs` puts ABC tokens and the boundary tokens into its target
sequence, and `_ar_losses` applies cross-entropy and optional KL to the complete
LM-head output. Our head slice is 32,769 classes and our loss positions start
at the first semantic prediction. ABC still supplies context and receives
indirect gradient through the transformer; it has no direct next-token target.

These choices were deliberate to isolate semantic adaptation and fit the
native prototype. Our independent gradient checks validate the chosen objective;
they do not establish parity with toolkit's different objective. Toolkit CE
numbers and KL weight 0.2 therefore cannot be compared directly to our means.

For the present remix objective, I would first add longer, correctly aligned
semantic sequences and real endings while retaining common-model planning.
An optional full-vocabulary/ABC loss is a later controlled experiment if we
want learned planning or closer toolkit replication. It needs memory profiling
and probably chunked readout losses. It does not inherently require a new GGML
operator. At inference, adapted planning should remain a separate evaluation.

## 4. Correct or remove the hybrid score-dropout curriculum

Toolkit dropout changes the instruction to symbolic-off as well as emptying
the ABC payload. Our paired full/dropout views preserve the full instruction.
Thus half of our real AR views teach a conditioning combination that is not
toolkit's off-mode path. The current UI always supplies a common-generated or
transcribed score, so true off-mode coverage is not an acceptance requirement.

For a score-first follow-up, test zero score dropout first and spend the saved
views on more genuine scored excerpts. If text-only/off mode becomes a goal,
prepare its actual instruction and prefix, and test that mode explicitly.

## 5. Test acoustic sampling and captions before changing many variables

The [scheduler's sigmoid branch](https://github.com/ostris/ai-toolkit/blob/73ebae13a468942ded8b47991b5d56eeeea25c36/toolkit/samplers/custom_flowmatch_sampler.py#L107-L143)
maps standard-normal draws through a sigmoid. About 72.8% fall between .25 and
.75; our uniform [.05,.95] sampler puts 55.6% there. This changes the acoustic
training emphasis. A matched NAR-only comparison can keep AR150, data, rank,
optimizer, update count and evaluation noise fixed while changing only this
distribution. Report per-sigma validation and listen on fresh content.

The [captioning preset](https://github.com/ostris/ai-toolkit/blob/73ebae13a468942ded8b47991b5d56eeeea25c36/ui/src/helpers/captionOptions.ts)
asks for specific instrumentation, production and musical descriptors. Our shared
caption can learn a common identity, but cannot associate different timbral
variants with distinct descriptions. Per-track or per-passage captions, retaining
`koan_palette`, are worth testing. They should describe audible events accurately;
automatic captioning is not a substitute for checking complex instrumental audio.

The inspected default trainer uses ordinary flow prediction error plus AR
auxiliary losses. Generic image/latent loss options and joint video/audio loss
branches are not evidence of a YuE2 decoded-waveform perceptual objective.
I found no default multi-resolution waveform/STFT or sound-design-specific loss
that we simply omitted. Our posterior-mean VAE target policy agrees with the
[toolkit VAE encoder](https://github.com/ostris/ai-toolkit/blob/73ebae13a468942ded8b47991b5d56eeeea25c36/extensions_built_in/audio_models/yue2/src/vae.py#L150-L162).

## 6. DoRA is a plausible independent experiment

Toolkit's [network selection](https://github.com/ostris/ai-toolkit/blob/73ebae13a468942ded8b47991b5d56eeeea25c36/toolkit/lora_special.py#L401-L405)
has generic DoRA support. Its [DoRA implementation](https://github.com/ostris/ai-toolkit/blob/73ebae13a468942ded8b47991b5d56eeeea25c36/toolkit/models/DoRA.py#L111-L150)
learns output magnitudes and detaches the normalization denominator. That capability
does not establish a verified YuE2 DoRA recipe, and the inspected Billie file
contains only LoRA factors, not magnitude vectors.

Native YuE2 currently trains additive A/B factors. SA3's `src/dit.h` already
implements magnitude scaling and a low-rank norm computation without forming a
dense adapted matrix. Reuse that machinery if pursuing YuE2 DoRA, while explicitly
choosing and checking the norm-gradient convention. Our independently verified
LoRA implementation does not automatically validate a new DoRA backward path.

A useful experiment would freeze AR150 and compare NAR LoRA against NAR DoRA
on the same improved data. At the current seven-projection layout, output-row
magnitudes add 573,440 trainable scalars per expert; runtime normalization still
needs a memory/throughput check. DoRA cannot by itself recover information absent
from the semantic labels. The user's SA3 preference motivates this test but is
not proof it will close the architecture/representation gap.

## Already retained, or not an obvious omission

- Both experts' relevant projections, teacher KL, separate expert learning rates,
  clean VAE targets, prompt conditioning, score transcription and fixed-code
  midpoint controls are already present in the native work.
- Toolkit explicitly detaches NAR's AR cache. Our staged optimization changes the
  optimization trajectory and keeps NAR conditioning fixed, but is not missing
  an essential NAR-to-AR gradient. Short alternating rounds are possible later;
  joint training is not automatically superior.
- Stem separation is optional and potentially useful for vocal isolation. It
  does not provide a ready-made decomposition of Koan's instrumental sound design.
- Fused QKV/gate-up adapters share A factors. Our independent projections are not
  equivalent under training, but are not a lower-capacity omission at rank 32.
  Current transformer/PEFT network code uses alpha equal to rank, matching our
  scale convention. The Billie run's exact historical scaling remains unproven.
- Toolkit's default optimizer/scheduler choices are not hidden evidence of an EMA
  recipe: the inspected config defaults to constant scheduling and EMA disabled.
  Its AdamW factory uses epsilon 1e-6; ours uses 1e-8. LoRA initialization also
  differs. These belong in a fidelity-to-toolkit ablation, not an uncontrolled
  collection of simultaneous changes. Quantized ConvRot training is primarily
  a memory strategy; we have not shown it improves sound design over our BF16 base.

## Upstream experience and the next decision

[Issue 1052](https://github.com/ostris/ai-toolkit/issues/1052) contains a firsthand
report of burnt high frequencies on cymbals and other noise-like sounds as small
sets are repeatedly trained. Its author gives a 15–34-song setup with rank 32,
60-second windows, KL and reduced AR learning rate. This resembles our listener
reports, but is neither an established cause nor a maintainer-endorsed fix.
The discussion also requests manually corrected ABC sidecars; our exact-prefix
and score-editing workflow is a useful capability to retain.

Recommended order:

1. Audit v4 versus v9 targets and preprocessing parity on the same Koan passages.
2. Expand distinct coverage, accurate captions and complete musical supervision;
   use the actual scored path instead of the hybrid dropout views.
3. Hold AR fixed and compare uniform versus sigmoid acoustic noise sampling.
4. Run a separate NAR DoRA comparison if target quality and coverage are adequate.

The present combined AR150/NAR400 model remains available for user experiments.
This review does not declare musical success or launch another fit. The strongest
remaining bottleneck may still be the reverse-engineered semantic representation;
the inspected toolkit uses the same head family rather than an unreleased official
encoder. Native GGML changes should follow a demonstrated resource/operator need,
not the assumption that toolkit has a fundamentally different differentiable core.

Raw reviewed sources, upstream discussions and file identities are archived in
`outputs/ai-toolkit-review-2026-10-25/`. A portable findings manifest accompanies
this document in `docs/experiments/yue2-ai-toolkit-review-2026-10-25.json`.
