# Two-track Koan semantic AR pilot

This is an exploratory test of real-audio semantic labels after the user
confirmed an audible palette change in the synthetic AR control. The user
reports that the Koan tokenizer reconstructions retain rhythm but are degraded,
and explicitly chose to try a small pilot despite that limitation. This is
not a target-quality pass by itself.

**Listening result:** the user reports that it is picking up the Koan palette
and retaining the supplied chords, with some degradation. By step 128 it loses
substantial melody. This is the first positive real-Koan palette/harmony result
in the presented comparison, not a blanket pass across all seeds and modes.
The user did not separately rate early-checkpoint melody fidelity, so chord
retention must not be relabeled as full melodic preservation. The
[feedback record](experiments/yue2-koan-ar-listening-2026-10-25.json) preserves
those distinctions. Step 32 remains the working reference; step 128 remains a
failure control.

## Data and training

| Split | Track | Source offset | Audio/token duration |
|---|---|---:|---:|
| Train | Cobalt | 16.24 s | 10.24 s |
| Train | Prism Pulse | 21.48 s | 10.24 s |
| Held out | Aeon | 20 s | 10.24 s |
| Held out | Silkwave | 20 s | 10.24 s |

Each track contributes one 256-frame excerpt. Train and held-out tracks are
disjoint by full-source hash. Tokens are exact crops of the earlier full-song
MotherSuperior extraction. The cached NAR K/V and genre-rich captions from
the previous NAR experiment are not reused. Source audio, codes, transcription
events, scores and prefix identities are retained in
`outputs/ar-koan-pilot-prep-2026-10-25/`.

Each example has a full-score prefix and a score-dropped view. The latter
removes the ABC interior while retaining the full-generation instruction,
lyric tags and boundary markers. Captions remain `Instrumental music.`.
The four training views are sampled uniformly with saved mt19937 state.
The fixed instrumental AR adapter is present during training and inference.

The run trains rank-8 factors in all 196 AR projections for 128 updates,
learning rate 5e-5, weight decay .01 and gradient clip 1. Frozen weights use
F16 storage and F32 projection compute. NAR, VAE, embeddings, norms and the
semantic LM head remain frozen. Crops have no EOS target. Training uses no
KL/replay regularizer; a known generated-music example supplies evaluation-only
retention loss. It does not contribute gradients.

The maximum prefix is 314 tokens; shorter views use masked left padding and
real-token positions starting at zero. Observed device memory peaks at
5,458 MiB, with 778.1 MiB of shared graph workspace, on the 8 GB laptop GPU.
No further GGML changes were required.

## Numerical verification

Padding/unpadding the same 50-token prefix preserves logits within 1.26e-6
relative L2 and adapter gradients within 1.19e-5. A separate CPU/PyTorch oracle
with 264 padding slots passes loss, logits, all factor gradients and the first
AdamW update. Multi-example resume reproduces the sample order and complete
optimizer-state file bit for bit.

The selected checkpoint also passes the independent gradient oracle with both
factors nonzero and a different actual prefix length. Its aggregate gradient
error is 1.24e-5, maximum significant-factor error 2.24e-5, and relative logit
error 2.84e-6. These are arithmetic checks, not listening-quality measures.

## Selection and overfitting

| Checkpoint | Training mean CE | Held-out mean CE | Retention CE |
|---|---:|---:|---:|
| Common model | 6.32094 | 7.07003 | 3.49975 |
| Selected step 32 | 5.84397 | 7.05411 | 3.50864 |
| Final step 128 | 1.54098 | 8.76378 | 3.59816 |

Step 32 is the minimum saved mean held-out loss across scored/dropout views.
Its improvement is only about **0.23%**, too small to treat as persuasive style
generalization. Later training strongly fits the two excerpts while worsening
held-out loss. The step-128 checkpoint is retained only as a labeled diagnostic
of stronger fitting versus degradation/content loss; it is not promoted as
the default adapter. Retention CE drift does not by itself measure content
adherence or distinguish intended style change from damage.

The score audit also reveals a significant limitation: Cobalt, Aeon and
Silkwave have no extracted note payloads in these short transcriptions; Prism
Pulse has 34. Chords, meter/structure and rhythm scaffolds can remain even
without note payloads. The scores therefore are not reliable melodic ground
truth. Aeon has changing meters, and its reported score-duration summary is
particularly uncertain. This pilot cannot validate accurate score/audio timing
or melody following. The held-out score-dropped losses independently reach
their minimum at step 32 as well; the selection is not driven solely by the
low-information scored views.

## Listening protocol

The local index is
`outputs/ar-koan-pilot-2026-10-25/auditions/listening.md`.
Eight primary blind A/B renders compare the common model and step 32:

- The full supplied 44.72-second piano/orchestral recording's score, at seeds
  831001 and 123.
- An independently authored 16-second score.
- A 16-second common-model plan originally generated from a neutral prompt
  without supplied audio or a reference score.

The planner stays on common weights. Exact score/prompt prefixes and flow seeds
match within each comparison; semantic streams are freshly sampled for each
AR model. No dataset recording or semantic-token stream is supplied at
inference. The user source is excluded as an exact training file, while its
underlying composition provenance is unknown. The authored score and generated
plan provide additional content controls.

The main decoder is native NAR, with fixed weights. A separate labeled pair
adds v9 to both AR conditions on the full user source, retaining identical
semantic tokens per AR condition. Two more labeled renders use step 128 on
the source and fresh score. Open these diagnostics after the blind comparisons.
That is 12 raw renders total, not 12 independent listening verdicts.

Listening exports are RMS matched with shared peak headroom within each group.
No EQ, time alignment or artifact-hiding edits are applied. Content guides are
listener references only. Judge absolute Koan palette resemblance, content
preservation and fidelity/degradation separately. Neither version matching the
dataset, or a stronger palette shift with worse degradation, are valid results.

Fixed decoder weights do not mean fixed conditioning: the adapted AR weights
change both sampled tokens and the AR states presented to NAR. These comparisons
test the complete AR-adapter effect, not an attribution solely to token IDs.

Portable numerical evidence:
[`experiments/yue2-koan-ar-pilot-2026-10-25.json`](experiments/yue2-koan-ar-pilot-2026-10-25.json).
The [target decision](experiments/yue2-ar-target-feedback-2026-10-25.json)
retains the user's qualification about degradation. Palette resemblance and
chord retention are positively reviewed in the presented comparison, with
degradation and step-128 melody loss also reported. Individual seed/mode ratings
remain unassigned. The [broader score audit](ar-score-supervision-2026-10-25.md)
prepares the next data-quality milestone without further fitting these two clips.
