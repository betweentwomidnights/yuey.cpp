# YuE2 adapter controls: future UI boundary

The October 26 listening assessment tentatively favors AR150/NAR400 for score
melody/chord adherence and AR200/NAR400 for Koan sound design. Text-to-music is
working better than remix. These are relative impressions, not a measured
adherence benchmark; coverage, context, captions and score dropout changed
together, so no individual setting has been established as the cause.

For a future gary4local integration or web trainer, keep ordinary generation
controls distinct from the training recipe. This document is design guidance;
the installer at `C:/dev/gary-localhost-installer` is not modified here.

| Surface | Useful controls | Meaning and limit |
| --- | --- | --- |
| Basic generation | Validated checkpoint pair, style prompt/trigger, seed, duration | Start with complete tested pairs; keep a base comparison available. |
| Advanced generation | Separate AR and NAR adapter strengths, temperature/repetition controls, decoder steps | AR affects arrangement and palette; NAR affects acoustic interpretation and can also alter audible musical content. There is no proven single melody-adherence slider. |
| Remix/content | Source transcription, editable score, optional score simplification | Correct score errors before treating them as an adapter failure. Keep source-as-composition separate from source-as-audio-reference. |
| Advanced trainer | AR/NAR ranks and learning rates, AR KL/replay, scored versus actual off-mode conditioning, coverage/window policy, LoRA versus DoRA | Save the whole recipe and model/data fingerprints with the checkpoint; label bundled experiments. |
| Expert trainer | Tokenizer-head/precision identity, noise-time distribution, DoRA norm-gradient convention, target vocabulary and genuine EOS policy | Use pinned defaults and diagnostics. These are not ordinary generation controls. |

Compatibility metadata should tie each acoustic checkpoint to its frozen AR
fingerprint, base checkpoint, head/preprocessing identity and required companion.
Changing AR conditioning is an experiment requiring evaluation, even if files
can technically be stacked. DoRA strength blends the complete normalized weight
delta; zero is exactly the base projection. Separate expert strength controls
would be useful, but are not implemented by this note.

Training selection should show score/content preservation and acoustic fidelity
separately, along with neutral retention and actual generation/remix auditions.
A lower semantic CE or flow loss is not automatically a better remix. The first
DoRA trial fixes AR150 and the original data so the acoustic adapter family is
the principal changed variable. The separate
[AR DoRA experiment](training-history.md#dora-comparisons-2026-10-26) tests semantic/style adaptation
with native NAR fixed. Its best validation loss is effectively tied with LoRA;
the user prefers AR DoRA in the matched listening comparison. AR DoRA must normalize around the frozen
instrumental companion and require its exact adapter prefix/strength at load.
The subsequent NAR100 calibration was rated about the same as native NAR;
NAR400 was rated slightly better in fidelity on the final source comparison.
The preferred candidate from these tests is AR DoRA150 plus ordinary NAR
LoRA400. This does not establish a broader score-adherence improvement.
The DoRA-capable experimental server now runs on port 8008 alongside the older
8007 server. All six Koan presets are labeled by AR family/update and acoustic
pair. Direct remix now has a seed field as well as create and score rendering.
For controlled exploration, keep an edited score fixed and change only the
preset. Separate AR/NAR strength controls remain future work.
