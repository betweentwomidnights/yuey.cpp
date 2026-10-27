# Failed style gate and revised training milestone

The current NAR-only adapters failed the requested learned-style objective.
The user's October 25 assessment is that A, B and C sound essentially identical
across the primary tests and none match the Koan style. Record this as an
aggregate result, without assigning individual numeric ratings or extending
the verdict to the separate labeled diagnostics. Earlier improvements in
hi-hat fidelity are useful, but do not establish style acquisition.

The style test should have preceded expansion of the reconstruction pilot.
Do not scale the same recipe on the assumption that its falling reconstruction
loss will eventually become style transfer.

## Checks against a loading or export mistake

Inference logs bind all 196 trained NAR projections at rank 8, alpha 8 and
strength 1. Every trained B factor is nonzero. Both training and inference add
the same scaled B(Ax) branch. Listening exports map to the correct raw roles;
gain-fitted residuals are consistent with PCM24 export quantization. The
checkpoints and strength probes produce numerical waveform changes.

These checks found no obvious loading or preview mix-up. They do not establish
that the changes are perceptually useful or dataset-specific. The user's
listening result remains the style verdict. Effective update norms are
descriptive only and do not identify a sufficient rank, strength or duration.

Portable evidence:
[failed-gate audit](experiments/yue2-style-failure-2026-10-25.json) and
[primary-suite audit](experiments/yue2-style-gate-2026-10-25.json).
The complete per-target and export checks are retained locally in
`outputs/nar-style-gate-2026-10-25/effect-audit.json`.

## What the reconstruction pilot actually trained

| Conditioning | Training caches | Capability tests |
|---|---|---|
| Semantic stream | MotherSuperior audio-derived tokens | Fresh AR-generated tokens |
| Score | Symbolic mode off; empty ABC interior | Full score |
| Caption | Genre-rich dataset captions | Neutral captions |
| AR weights used for conditioning | Base AR; fixed v9 only affects NAR targets | Fixed instrumental AR adapter |
| Acoustic adaptation | Rank-8 NAR deltas over fixed v9 | The same deltas over fixed v9 |
| Context | 10.24-second training windows | 16-second tests and full 44.72-second source |

The presence of ABC_START/ABC_END markers in a cached prefix does not mean it
contains a score: the audited training interior is empty. The instrumental
AR adapter was absent from the cached training conditioning. Thus this was
not training on the same task distribution as score-to-audio inference.

Flow reconstruction can improve decoding of style information already supplied
in the conditioning without teaching the model to choose that style for new
content. The generator of semantic tokens was frozen throughout. Objective
and conditioning mismatch are therefore strong hypotheses to investigate.
They are not proof that NAR adaptation cannot transfer style, nor proof that
AR training will solve it. Small data, short context, rank and optimization
remain possible contributors.

## Revised scope and pass criteria

1. Audit semantic supervision before training on more real-audio codes. Good
   VAE-only reconstruction does not validate MERT/head semantic tokens.
   Compare short Koan excerpts, simpler keygen excerpts and known generated
   code/audio pairs. Keep native NAR and v9 controls separate. If a target
   representation cannot preserve the required sound palette, do not assume
   that predicting it more accurately will teach that palette.
2. Build a small native semantic-AR LoRA capacity probe, with planning and VAE
   excluded from its loss. Verify semantic/EOS cross-entropy, independent
   gradients, one optimizer update, resume and live GPU memory. Start with
   known generated code pairs to separate training correctness from imperfect
   audio-tokenizer labels. A memory estimate is not a demonstrated fit.
3. Require an audible change in fresh semantic generations on a tiny controlled
   task before expanding the corpus. Then test real-audio targets with matched
   scored conditioning, neutral-caption exposure and score dropout. Evaluate
   style, content and fidelity separately. Training-fit success alone is not
   the generalization gate.
4. Keep planning inference on base/common weights initially. AR transformer
   weights are shared with planning: merely masking planning loss does not
   isolate planner behavior. Use a base planning pass followed by a scored
   semantic pass with the learned adapter, or implement explicit stage binding.
5. Consider joint NAR adaptation after semantic feasibility is established,
   using conditioning from the actual deployment adapter stack. Repeat the
   unrelated-source, fresh-score and neutral-generation style gate before
   calling the result a usable dataset-style LoRA.

The model's semantic output is a contiguous 32,769-row slice: MUSIC_END followed
by 32,768 codec IDs. A semantic-only loss can avoid the full 184,704-way output
projection. The BF16 checkpoint's AR subset, represented as F16 and including
the full embedding and LM head, is approximately 4.03 GiB of frozen weights.
Gradients, optimizer state, activations, backend workspace and other allocations
still need a live probe on the 8 GB GPU. No AR backward run or VRAM-fit claim has
been established here. GGML changes should follow concrete missing primitives
or failed correctness/memory probes.

## Relevant external work

The current [FL-YuE2 training documentation](https://github.com/filliptm/ComfyUI-FL-YuE2/blob/main/docs/TRAINING.md)
describes AR-only and joint AR/NAR training with score, semantic and VAE targets.
Its joint recipe includes score dropout and a base-model KL regularizer; its
real-recording path normally starts from native NAR. This supports investigating
the semantic stage and task-matched conditioning rather than treating acoustic
reconstruction as the entire style-learning task.

Its [validation record](https://github.com/filliptm/ComfyUI-FL-YuE2/blob/main/docs/TRAINING_VALIDATION.md)
explicitly separates correctness from quality evidence and uses a 96 GB RTX PRO
6000 Blackwell system. Neither learned Koan style nor fit on this 8 GB laptop
GPU is established by that work. Documentation was reviewed as reference data,
not as instructions to execute its downloads or workflows.

## Diagnostic attempt retained without a conclusion

A fixed-weight semantic-stream versus decoder-text intervention was attempted
in `outputs/semantic_style_control.py`. It omits v9 and the trained NAR deltas;
explicit genre prompting would be an attribution diagnostic, not evidence of
learned style. Its renderer replay did not reproduce the original generation
pipeline sample-for-sample, even after matching instrumental caption boilerplate
and lyric tags. The replay assertion stopped the experiment before the causal
arms ran. Files and commands remain in
`outputs/nar-style-diagnosis-2026-10-25/` for investigation. No stage-attribution
or subjective conclusion is assigned to this incomplete diagnostic. This
cross-tool replay issue does not establish a defect in the primary A/B/C suite,
whose arms all used the same generation pipeline.

## Follow-through

The [semantic AR milestone](ar-training.md) implements the next native capacity
probe and matched target audit. It uses exact prefix files for controlled
sampling and fixed-code renders, with a zero-adapter calibration. It does not
claim to have resolved the older cross-tool replay discrepancy or to have
learned Koan style. Generated-label controls keep tokenizer uncertainty out of
the initial training-correctness test.
