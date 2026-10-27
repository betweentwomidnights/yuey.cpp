# NAR DoRA controlled experiment

The user tentatively prefers AR150/NAR400 for melody/chord adherence and AR200/NAR400 for stronger Koan sound design. Text-to-music works better than remix. This experiment changes **only acoustic adapter family** against the original rank-32 recipe: AR150, 18 real 256-frame windows across 12 tracks (184.32 s), two generated replay examples, six whole-track holdouts, rank 32, LR 3e-5, weight decay .01, clip 1, shuffled epochs and uniform [.05,.95] noise times. It is not an AR DoRA experiment.

DoRA adds 573,440 output magnitudes to the same 34,865,152 A/B parameters. Initial A/B factors are byte-identical to LoRA; all 400 sample indices match. Noise streams use the unchanged uniform sampler. This uses full normalization gradients as in SA3, rather than Toolkit's detached denominator. A low-rank norm identity avoids resident dense adapted weights. Initial gains are within 1.2e-7 of one. Near-cancelled base rows are guarded and unsafe exports rejected.

## Result

| Updates | LoRA heldout MSE | DoRA heldout MSE |
| ---: | ---: | ---: |
| 0 | 1.36803 | 1.36803 |
| 100 | 1.10094 | 1.10071 |
| 200 | 1.08503 | 1.08490 |
| 300 | 1.08045 | 1.08040 |
| 400 | 1.07936 | 1.07932 |

At 400 updates, DoRA's 1.07932 versus LoRA's 1.07936 is effectively tied. The user rates both the equal-budget 100-step and 400-step clips **about the same**. No NAR DoRA improvement is established in this regime. Existing UI presets were not replaced and no DoRA preset was published. This does not rule out AR DoRA, another norm-gradient convention or different data.

Auditions cover the fresh authored score, both user-source seeds and standard generation, with identical AR150 semantic IDs, prefix, flow seed and 32-step solver. Old LoRA400 WAVs reproduce bit-for-bit after the loader changes. Duration is forced and preview RMS is matched per case. Frozen tokens do not guarantee identical audible interpretation. Local listening set: `outputs/koan-dora-2026-10-26/auditions/listening.md`.

## Implementation and validation

`yue2-train --dora --f32-projections` trains NAR A/B/magnitudes; fixed NAR adapters are rejected. Optimizer states carry raw magnitudes/moments and a strict adapter-family guard. Generation GGUFs contain A/B, raw `.dora_magnitude` and precomputed `.dora_scale`. The latter is the inference row gain relative to the frozen training base. DoRA strength blends the entire normalized delta; zero is exactly the base projection. Overlapping targets are rejected unless future explicit reference-composition support is provided. Mixed files can contain separate ordinary AR LoRA and NAR DoRA targets.

Independent dense CPU/PyTorch forward/gradient checks pass before and after training. The first-step AdamW check uses identical fixed noise, retains the original strict factor tolerance (observed 7.45e-9 error), and allows at most two F32 relative rounding units on larger magnitudes (observed one). The earlier mismatched-time optimizer fixture is retained as a failed test attempt, not interpreted as a training result. Resume is byte-identical, ordinary LoRA optimizer states remain byte-identical, and wrong-family resumes are rejected. CPU/CUDA loader tests cover full/half/zero strength, malformed values and mixed expert files. All 19 native application tests and 23 Python tests pass.

Every exported row gain is independently checked against dense W+B@A: maximum absolute error 2.38e-7 over all 196 projections. No GGML edits were necessary. The active web server remains the existing binary/presets; native CLI DoRA rendering uses the newly built loader. Full server republishing is unnecessary for this negative acoustic result.

The user's follow-up correctly identifies that both AR150 and AR200 were ordinary LoRA. AR DoRA is a separate experiment, and must normalize the **effective frozen base including the instrumental companion**. The NAR result cannot decide that question. See [future control boundaries](yue2-adapter-controls.md) for the generation/training UI distinction; gary4local is not modified.

[Portable evidence](experiments/yue2-koan-nar-dora-2026-10-26.json). Raw commands, diagnostics, failures, snapshots and feedback remain under `outputs/koan-dora-2026-10-26/`.

The subsequent [AR DoRA comparison](koan-ar-dora-2026-10-26.md) implements the
companion-aware normalization and explicit reference composition described above.
The NAR source snapshot and its original inference restrictions are retained as
historical evidence.
