# Dataset-style capability gate

This experiment implements the user's
[clarified style-transfer objective](yue2-style-transfer-objective.md). It runs
existing adapters on fresh content and neutral captions. **No additional
training was performed.** Successful execution is not a style-transfer verdict.

**Result: failed on 2026-10-25.** The user reports that A, B and C sound
essentially identical across the tests and none match the Koan style. This is
an aggregate judgment across the six primary groups, not invented per-case
scores. The labeled strength and native-decoder controls have no reported
subjective verdict. Prior fidelity preferences do not override this result.
See the [diagnosis and revised milestone](nar-style-diagnosis-2026-10-25.md).

## Inputs and inference modes

| Mode | Content | Coverage |
|---|---|---|
| User source to score | The supplied piano/orchestral recording | Full 44.7187 seconds, two seeds |
| New score | Independently authored C-major 3/4 theme | Eight bars / 16 seconds |
| New score | Independently authored A-minor syncopated theme | Eight bars / 16 seconds |
| Standard generation | Neutral instrumental prompt, model-generated plan | Eight bars / 16 seconds |
| Standard generation | Neutral dynamics prompt, model-generated plan | Eight bars / 16 seconds |

The main source is the user-supplied `gary4juce_1792943821821.wav`, SHA-256
`6764c79b45b9812269a91505d1f01653909fdb71b87c407fab442aa12ec5cbb6`.
It is copied without modification, and the original remains unchanged. It is
not an exact file from either split of the expanded pilot; its underlying
composition's training provenance is not known. The independently authored
scores and model-generated plans provide additional content controls.

The transcriber reads the recording as 32 bars of 2/4 at 86 BPM in C minor.
Thirty-two half-length bars are duration-equivalent to the user's sixteen
4/4 bars. The entire recording is retained, with 1,118 semantic frames and
2,146,496 decoded audio frames. The generated ABC, MIDI and transcription
events are retained for investigating content errors.

Only `Instrumental music.` and `Instrumental music with changing dynamics.`
are used as production captions. No artist name, target genre description,
training-track caption or dataset recording is supplied to generation. The
user recording supplies content through transcription. No source VAE latents
or audio-derived semantic-token stream is supplied to the decoder.

The authored-score pages include dry synthesized melody guides for the
listener. These guides are not model inputs. The standard-generation cases
have no supplied audio, score or ABC prefix; the model generates its own
plan from the neutral prompt and structural meter/tempo/key settings.

## Models and controls

The six primary blind groups compare:

1. Fixed v9 acoustic companion, with the common instrumental AR adapter.
2. The same fixed companions plus the original four-track step48 delta.
3. The same fixed companions plus expanded twelve-track step192.

All six permutations of these models behind A/B/C are used once, so each
model appears twice under each letter. Used scores and semantic streams match
exactly within each group. Holding this content fixed isolates the acoustic
effect of NAR adaptation; it does not test a changed AR sampling distribution.

Two separate labeled probes double the expanded delta's strength on the user
source and one standard-generation case. They test whether increased strength
produces style, only changes fidelity, or damages musical content. They are
not included as another primary model or automatically selected as winners.

Two further labeled controls omit v9 **and all learned NAR deltas**, retaining
the same instrumental AR adapter. These controls use the user source and one
standard-generation case. They also retain exactly the same used score and AR
semantic stream as their fixed-v9 comparisons.

This last control matters: previous comparisons kept v9 fixed because the
learned deltas were trained on top of it. Fresh AR-token inference also has a
native decoder path without that companion. A benefit over fixed v9 alone
could reflect companion compensation rather than acquisition of a particular
dataset sound. The native-decoder control must therefore be considered when
interpreting the learned adapters. No conclusion about its quality is assigned
without listening.

## Execution and review

All **22 renders** completed on the existing GGML 0.27 CUDA build: 18 primary
renders, two strength probes and two no-v9 controls. Every within-group plan
and semantic stream check passed. All exported listening files have finite,
non-silent samples and peak headroom. Each group is RMS matched with a shared
peak-prevention gain. Audio is not EQ processed, time aligned or edited to
hide artifacts.

The local index is `outputs/nar-style-gate-2026-10-25/listening.md`. Start with
`source-audio.md`, then `fresh-scores.md` and `standard-generation.md`.
`strength-diagnostic.md` and `native-nar-control.md` are labeled diagnostics;
use them after the blind comparisons. The separate answer key remains local.

Judge three questions independently:

- Does any version convincingly resemble the Koan dataset's instrumentation,
  bass/sound-design textures, percussion production and spatial character?
  A relative favorite alone is insufficient: **none match** is a valid result.
- Does it retain the source or score's intended musical content? For free
  generation, judge musical coherence and the neutral request instead.
- Which has the best fidelity and fewest artifacts?

Cleaner hi-hats alone do not establish style transfer. The decision gate is a
dataset-specific shift on new content and standard generation, with acceptable
content adherence and fidelity. The primary style gate failed; separate
content and fidelity ratings remain unassigned.

## Limits and interpretation

The neutral captions intentionally differ from the genre-rich training
captions. A failure may indicate a conditioning/training-protocol problem,
not prove that NAR adaptation is incapable of style transfer. The full user
source also exceeds the 10.24-second training-window length; context-length
generalization is another possible limit. The AR stage remains frozen, and
style expressed through its semantic sequence may require another adaptation
strategy. Additional steps on the same NAR objective are not assumed to solve
these limits.

Transcription may omit performance, polyphony or musical detail; it is visible
as a separate content stage. This test does not evaluate source-latent warm
starts. Explicit semantic budgets bound the comparisons and do not establish
natural full-song endings.

`tools/evaluate_nar_style.py` produces the primary suite and strength probes.
Its exact source is archived with the evidence and checked against the saved
hash. The local `outputs/style_gate_plain_nar_control.py` adds the no-v9
controls by removing only the acoustic companion from otherwise identical
baseline commands. Commands, hashes, raw float WAVs, plans, codes and preview
audio are retained. Runtime/model identity changes require a fresh output
directory; existing incomplete renders are not silently overwritten.

The portable audit is
[`experiments/yue2-style-gate-2026-10-25.json`](experiments/yue2-style-gate-2026-10-25.json).
