# YuE2 LoRA objective and the next evaluation gate

The user clarified the intended outcome on 2026-10-25: a reusable adapter that
makes **new scores, unrelated input audio, and standard generation sound like
the training dataset**, without supplying dataset reference tracks at inference.
The dataset supplies the learned style through model weights. New input supplies
the musical content. Faithfulness to that content and audio fidelity also matter.

**Prompting clarification, 2026-10-25:** the user explicitly accepts Koan-related
style descriptions and learned trigger words as part of a successful remix
workflow, as with their SA3 and ACE-Step adapters. Neutral prompting is optional
diagnosis, not a production acceptance requirement. Compare the same style
prompt with and without the adapter to isolate its contribution. The requirement
to work without dataset reference recordings at inference remains. A proposed
new trigger must be taught; the existing checkpoints do not establish an
association with a newly invented alias. See the
[policy record](experiments/yue2-prompted-style-policy-2026-10-25.json).

The current target is the Koan Sound training subset's sound palette,
instrumentation, sound-design textures and production character. Cleaner
high-frequency rendering alone does not establish success at this objective.
The prior fidelity experiments were feasibility and regression gates; they
must not become the sole optimization target for the project.

Current listening update: the real-Koan semantic AR pilot now has positive
palette and chord-retention feedback on the presented input-score comparison,
with some degradation. Step 128 loses substantial melody. This advances the
reference-free style objective in that example; it does not establish full
melody preservation or success across all inference modes. Keep step 32 as the
working comparison reference and step 128 as a failure control. The
[expanded score-supervision audit](ar-score-supervision-2026-10-25.md) identifies
candidate data for broader, melody-informed training rather than extending
the two-excerpt run.

## What the current evidence establishes

Native GGML NAR training works, matches independent gradient references and
produces useful acoustic changes. The user repeatedly preferred cleaner
rendering, initially against fixed v9 and then in several expanded-versus-
original comparisons. The expanded seed-831001 review has five preferences for
step192, one for original step48 and two ties, one leaning expanded. Clip 04
was explicitly corrected to B, which is original step48.

The NAR-only adapters **failed the October 25 neutral-caption capability gate**.
The user reports that A, B and C sound essentially identical across the tests
and none match the Koan style. Earlier reductions in degradation remain
fidelity evidence only. The [diagnosis](nar-style-diagnosis-2026-10-25.md)
records conditioning mismatches and the proposed semantic-training milestone.
That result does not rule out prompted use under the user's subsequently
clarified goal, and it is separate from the later positive AR-pilot feedback.

The listening tracks were excluded from training, including two keygen examples
outside the Koan pool. Their transcribed scores and source captions nevertheless
condition the tests. Captions explicitly describe genre/tempo/key, and the base
model contributes its own style. Holding AR tokens identical between NAR
checkpoints isolates acoustic effects, but cannot show a learned change to the
AR/semantic sampling distribution. These tests neither require training tracks
at inference nor prove the requested reference-free style capability.

## Completed capability evaluation

Further scaling of the same reconstruction/fidelity experiment is paused.
The completed gate explicitly tested the user's desired inference modes:

1. Independently authored scores, not transcribed from the training or current
   validation tracks, using neutral captions. Compare fixed v9, original step48
   and expanded step192 with the same content and seeds.
2. Genuinely unrelated input audio as content. Evaluate source-to-score remix
   separately from any VAE-latent warm-start mode so source preservation and
   style change remain distinguishable. No dataset recording is supplied as a
   style reference to either path.
3. Standard generation from neutral prompts, without a supplied source recording
   or supplied reference-derived score. The model can generate its own plan;
   this historical test used neutral captions to isolate the learned effect.
   The current goal also permits descriptive prompts and triggers; use an
   identical-prompt adapter-off control before attributing their effect to
   the learned weights.

Use separate judgments for **dataset-style resemblance**, **musical-content
adherence**, and **audio quality/artifacts**. In particular, do not count a
cleaner hi-hat as proof of learning the dataset's bass textures, instrumentation,
percussion production or broader musical vocabulary. Listener style anchors,
if used, are evaluation references and are never model inputs.

NAR-only training remains a hypothesis for transferring sound palette and
production character. The frozen AR/semantic stage may constrain the available
style changes. If it consistently produces a cleaner version of the same sound
without the desired stylistic shift, revise the training/conditioning design
and assess whether semantic/AR adaptation is needed. More NAR steps alone are
not a demonstrated route to the user's goal. No promise is made that the
current adapter can deliver the full requested style transfer.

After the user authorized the next milestone and supplied a piano/orchestral
recording, the [style-capability gate](nar-style-gate-2026-10-25.md) was executed
without further training. It includes the full source, fresh authored scores,
standard generation, strength probes and native-decoder controls. Its audio is
reviewed for style: it failed. There is no demonstrated learned Koan-style
adapter to promote. Separate content/fidelity ratings and the labeled
diagnostics remain unassigned. The next milestone is a small semantic-stage
training feasibility and target-quality investigation, rather than more
updates on the unchanged NAR reconstruction recipe.

The subsequent [semantic AR capacity milestone](ar-training.md) implements and
checks that native path using known generated token labels, plus a synthetic
electronic-target control under neutral inference captions. It also audits
real-audio semantic targets with native/v9 decoder controls. These numerical
and listening controls do not change the failed Koan-style verdict: neither
capacity adapter has trained on Koan audio, and perceptual style/target-quality
judgments remain separate gates before corpus expansion.

The user then confirmed an audible palette change in the presented synthetic
unseen-score comparison, without assigning target resemblance or content/fidelity
ratings. The real-audio reconstructions were judged rhythmically useful but
degraded; the user explicitly authorized a small pilot despite those concerns.
The [two-track Koan AR pilot](koan-ar-pilot-2026-10-25.md) proceeds on that basis,
without converting authorization into a target-quality pass. Its early
checkpoint and overfit diagnostic have now received partial review: Koan palette
and chords are retained, some degradation remains, and the late checkpoint
loses melody. Unreported cases and dimensions remain unassigned.

The user accepts descriptive Koan prompts and learned trigger words. The
[rank-32 follow-up](koan-rank32-2026-10-25.md) teaches `koan_palette` with broader
track coverage, KL/replay retention and matched NAR calibration. On its initial
full-source seed-831001 AR-only preview, the user prefers rank-32 step 150's
balance of palette and melody over the previous rank-8 step 32 under identical
prompts. This does not assign ratings to the additional cases or the new NAR stage.
