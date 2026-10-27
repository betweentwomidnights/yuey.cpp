# Koan AR DoRA web UI rollout

The listener-preferred AR DoRA150 plus calibrated NAR LoRA400 is available at
**http://127.0.0.1:8008/**, together with AR DoRA150/native NAR and all four older
AR LoRA150/200 presets. The dropdown labels distinguish the AR adapter family
and the matched acoustic model. The four earlier export files are unchanged.
No new training or GGML changes were needed.

The existing 8007 server remains the older binary and cannot load DoRA.
Automatic approval review rejected restarting that process, reporting only
“blocked by policy.” A separately built runtime on port 8008 completes the
rollout without stopping it. Use 8008 for these comparisons. The native CLI
default port is unchanged.

## Using the presets

Select **Koan · AR DoRA 150 + NAR LoRA 400** for the preferred candidate or
**Koan · AR DoRA 150 + native NAR** to compare acoustic calibration. The other
four labels expose AR LoRA150 and AR LoRA200 with their own native/calibrated
NAR pairs. Each NAR400 is matched to a different frozen AR. A saved selection
is retained; the code's previous fresh-browser fallback is unchanged.

Use `koan_palette` in the prompt (or the Koan prompt button), keep instrumental
mode and its common companion enabled, and begin at strength 1, temperature 1
and 32 decoder steps. The combined strength scales both experts together;
separate AR/NAR strengths remain future work.

Create, direct remix and edited-score rendering each expose a seed field.
Direct remix previously omitted the seed and therefore always requested a
random one; its new field is forwarded by the shared remix/continuation
controls. -1 remains random. For close comparisons, transcribe once, keep the
edited score fixed, set its render seed (for example 831001), and switch only
the preset. Keep prompt, strength, temperature, decoder steps, encoding and
audio ceiling fixed. A common seed does not force the AR models to sample
the same semantic tokens, and a ceiling is an upper bound rather than an
identical-duration guarantee.

## Export and verification

The AR-only alias is a byte-identical copy of the selected DoRA checkpoint.
The combined export is `mixed_lora_dora`: AR A/B/magnitudes/row gains plus the
ordinary NAR A/B factors, 1,176 tensors. Every tensor is byte-identical to its
source. The frozen instrumental companion reference fingerprint is preserved.
The common companion must precede the style adapter at its training strength;
normalization includes that companion. Planning stays on base/common weights.

Fresh-score CLI generation with either UI export reproduces the previously
auditioned token stream and WAV bit-for-bit. Live CUDA API checks verify all
six presets generate, the same input score remains fixed across presets,
zero strength is audio-identical to the common model, and repeated fixed-seed
generation is audio-identical. Two repeated four-second source remix requests
produce identical transcription scores and audio. Selected adapter/strength and
seed metadata are checked. These are short plumbing checks (64 semantic frames,
8 solver steps), not additional perceptual evaluations.

Four focused native tests pass: loader, DoRA normalization, server support and
server progress. Browser checks show all six labels, the preferred pair and a
remix seed of 831001 retained across preset changes. Screenshots and immutable
source snapshots are saved with the rollout. The earlier CPU-cache numerical
limitation remains documented in the acoustic calibration report; publication
does not remove it.

Runtime: `outputs/koan-ar-dora-webui-2026-10-26/runtime/yue2-server.exe`.
Launch from the repository root with `--models-dir models --adapters-dir models
--encoding Q4_K_M --device cuda --host 127.0.0.1 --port 8008` and
`GGML_CUDA_TF32=0`. PID/logs are under the same rollout directory. The current
instance releases models between jobs.

[Portable evidence](experiments/yue2-koan-ar-dora-webui-2026-10-26.json),
[server usage](server.md), and
[acoustic calibration](koan-ar-dora-nar-calibration-2026-10-26.md).
