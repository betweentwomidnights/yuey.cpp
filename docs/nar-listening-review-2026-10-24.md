# Native NAR LoRA listening review

The user reviewed seed 831001 and the three longer comparisons before the
assistant decoded their A/B identities. The four-track step-48 adapter was
preferred clearly on six short clips. The two remaining short clips were
near ties with a lean toward that adapter. No reported short-clip preference
favored the fixed-v9 baseline.

## Decoded feedback

| Case | Learned step-48 label | User assessment |
|---|---|---|
| Clip 01 | A | Essentially identical; tentative A fidelity lean. Recorded as a tie with an A lean. |
| Clip 02 | A | B has degraded high end; both have a jarring intro. A preference is inferred from the description. |
| Clip 03 | A | A clearly higher quality. |
| Clip 04 | B | Very close; B may be slightly better. |
| Clip 05 | B | B clearly better; A very degraded. |
| Clip 06 | B | B fuller and higher quality; A degraded. |
| Clip 07 | B | B fuller; A has degraded/squeaky high end. |
| Clip 08 | A | A fuller; B has degraded high end. |
| Long 30 s | A | A cleaner, with a small difference. |
| Long 60 s | B | Hard to distinguish; recorded as a tie. |
| Long 120 s | A | Hard to distinguish; recorded as a tie. |

## Seed 123 spot-check

The user then reviewed clips 03, 05, 07 and 08 at seed 123. In this selected
subset the learned adapter happens to be B in all four pairs.

| Case | Learned step-48 label | User assessment |
|---|---|---|
| Clip 03 | B | B preferred; the opening crash cymbal is squeakier in A. Both outputs are otherwise appealing. |
| Clip 05 | B | Both have a degraded/bandpass-like drum character; no preference stated. |
| Clip 07 | B | Tentative B fidelity preference; very close. |
| Clip 08 | B | A has degraded high end; B sounds a bit better. |

Three directional preferences again favor step 48, with varying strength. Do
not count them as three decisive wins. Clip 05 remains without an assigned
preference; shared degradation is not automatically a tie or an adapter loss.
The bandpass-like impression alone does not establish intentional filtering.

The letter B alone is not balanced within this subset. However, the previous
seed assigned the learned adapter to A on clips 03 and 08, and the preferences
on those tracks follow the adapter across that label switch. This supports
the direction of the original fidelity result with another semantic stream.
It is still one listener on repeated source tracks, not four new sources.

The label reversal on clip 08 follows the learned adapter, rather than a
consistent preference for the letter B. Both keygen examples also favor the
learned adapter, although its training windows came from Koan tracks.

These judgments support improved acoustic rendering, especially fewer
high-frequency artifacts and greater perceived fullness. The A/B files were
RMS matched, and semantic streams match within every pair. The decoder adapter
is the experimental difference. That makes this evidence for the NAR update's
effect in this workflow. It does not yet isolate acquisition of a particular
artist's timbre or sound-design vocabulary: captions and the pretrained model
also contribute style to both variants.

## Relationship to the source

These auditions use:

```text
source excerpt -> ABC transcription + source caption
               -> fresh AR semantic tokens
               -> NAR with fixed v9, with or without learned step-48 delta
               -> unchanged VAE decoder
```

The source waveform, source VAE latents and semantic tokens predicted from
source audio are not supplied to this generation path. The ABC score conveys
the notes, chords, timing and tempo that transcription captures, followed by
the existing instrumental score handling. It does not encode the complete
recording, percussion, performance, mix or intricate sound design. The AR
stage generates a new semantic stream from that symbolic conditioning.

The user's observation that the clips are useful, dataset-influenced audio
without sounding identical to their sources is therefore consistent with this
resynthesis workflow. Source faithfulness remains a separate question. If
melodic or harmonic identity is also missing, transcription accuracy and AR
score adherence need their own evaluation. Better NAR sound quality does not
establish those properties or resolve the audio-to-semantic tokenizer problem.

The jarring intro shared by both versions of clip 02 is a shared-path issue
to investigate separately; these ratings do not implicate the learned delta.
The long-render ties establish no clear perceptual benefit at those lengths;
they also report no audible regression in the comparison.

## Next decision

Keep step 48 as the reference adapter. The seed-123 spot-check supports a
bounded expansion of the training pilot; further listening on every existing
pair need not block that experiment. Seed 2026 and the remaining four seed-123
clips remain available and unrated.

The next training pilot should increase unique
audio exposure across more training tracks and multiple positions per track,
while preserving all current whole-song holdouts. Keep rank, frozen AR/head/VAE,
fixed v9 and window length unchanged initially. Train a fresh dataset-identified
run, and compare its checkpoints against both v9 and the preserved original
step 48. Do not present these reviews as proof that more optimizer steps on the
original four windows will help.

For audio-preserving remix, evaluate the existing source-latent strength
control separately. It addresses a different source-preservation tradeoff from
the score-to-fresh-semantic-code path reviewed here.

This is one reviewer's assessment of all seed-831001 pairs and a partial
second-seed spot-check. Two short tracks were previous
validation tracks; four are new Koan holdouts and two are keygen examples.
All long comparisons use the same track as short clip 06, so they are not
additional independent source tracks. No score is assigned to unreviewed pairs.

The machine-readable feedback is in
[`experiments/yue2-listening-feedback-2026-10-24.json`](experiments/yue2-listening-feedback-2026-10-24.json).
The subsequent seed-123 feedback is in
[`experiments/yue2-listening-feedback-seed123-2026-10-24.json`](experiments/yue2-listening-feedback-seed123-2026-10-24.json).
The local listening folder retains `ratings.json`, the separate answer key,
and the original audio without changes.

The resulting [expanded 12-track pilot](nar-expanded-pilot-2026-10-24.md) is
now complete. Its new A/B set compares the expanded candidate against the
original step-48 adapter; the feedback above applies to the earlier v9-versus-
step-48 experiment, not to those new comparisons.
