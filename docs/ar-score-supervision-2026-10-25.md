# Broader score-supervision audit after the listening result

The user reports Koan palette acquisition with chord retention, some degradation,
and substantial melody loss at step 128. Keep the early step-32 adapter unchanged
as the working reference. Further updates on the same two excerpts are not the
next experiment. The original pilot supplied almost no explicit note supervision
in one training example, and its later updates visibly overfit held-out labels.
Sparse score supervision is a plausible contributor to content drift, not an
established sole cause.

This follow-up audits existing feature windows only. **No new LoRA training was
performed.** It selects the middle two predetermined windows from each of the
12 training tracks, plus the existing window from all six whole-track holdouts.
Selection is independent of model validation loss. Dataset audio/code windows
and the step-32/128 adapters are unchanged.

| Split | Windows / tracks | Zero-note windows | Windows flagged for review | Tracks represented among candidates |
|---|---:|---:|---:|---:|
| Training | 24 / 12 | 1 | 18 | 12 |
| Held out | 6 / 6 | 2 | 2 | 2 |

The triage flags an excerpt with at least eight note payloads, three distinct
pitches and notes covering 25% of the excerpt. These thresholds identify review
candidates; they are not validated transcription-quality criteria. The 18
training candidates span all 12 training tracks and 184.32 seconds of unique
audio, nine times the two-excerpt pilot. Pitch/onset accuracy and score/audio
alignment remain unverified. A zero-note excerpt may still contain useful
chords, rhythm and timbre information, so it is not automatically bad style data.

Source audio/code hashes, transcription events, score hashes, pitch diversity
and time coverage are recorded in
[`experiments/yue2-ar-score-coverage-2026-10-25.json`](experiments/yue2-ar-score-coverage-2026-10-25.json).
The local table and raw transcriptions are in
`outputs/ar-score-coverage-2026-10-25/index.md`. All six held-out tracks are
preserved, including the weak-score examples; their limitations should be
reported separately rather than choosing favorable validation tracks.

## Next-experiment contract

1. Review the candidate score/audio pairs and expand AR training with a
   consistently taught trigger, useful style descriptions and matched
   instrumental conditioning, as the user now explicitly permits. Neutral
   captions remain optional controls. Limit repeated exposure per
   excerpt and keep early checkpoints. Compare against immutable step 32, using
   palette, melody/chords and degradation as separate outcomes.
2. Use a fixed set of unrelated scores and common-model generations to monitor
   content drift. Current retention CE is monitoring only. A teacher-retention
   objective would need implementation and an independent gradient check; it
   is not already active and is not guaranteed to protect melody.
3. Investigate degradation separately with the chosen AR adapter frozen. A fresh
   NAR calibration would need conditioning prepared with that exact AR stack
   and high-quality real-audio VAE posterior targets. Reusing old NAR deltas
   trained under different conditioning is not a controlled fidelity test.

Do not change AR and decoder weights simultaneously in the main comparison.
The cause of degradation remains unresolved: imperfect semantic labels,
conditioning changes, decoder behavior and sampling may contribute. Neither
data expansion nor NAR calibration is a demonstrated repair yet.

The [follow-up manifest](experiments/yue2-ar-followup-2026-10-25.json) records the
candidate windows, held-out membership, reference checkpoint hash and proposed
controls. It is a data/experiment contract, not a completed training run or a
claim that the candidate scores are correct.

The [sv-billie adapter review](sv-billie-review-2026-10-25.md) inspects an actual
dual-expert rank-32 release and updates the prompted evaluation contract. Its
training settings are only partly public; current toolkit support for retention
and joint objectives should not be mistaken for that adapter's exact recipe.
