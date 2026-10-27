# Koan AR training data audit and final expansion scope

The three useful UI pairings do not yet compare full-dataset recipes. AR200
changes data coverage, context length, captions and score dropout together.
The user's latest assessment favors AR200 LoRA for remix palette, while all
three pairs produce useful text-to-music. This does not identify LoRA or DoRA
as universally better; the earlier matched AR150 listening preference for DoRA
remains a separate, narrower result.

## What actually trained

| AR adapter | Real training tracks | Distinct real windows | Window duration | Unique real training audio | Captions |
| --- | ---: | ---: | ---: | ---: | --- |
| LoRA150 | 12 | 18 | 10.24 s | 184.32 s / 3.07 min | One shared Koan caption |
| DoRA150 | 12 | 18 | 10.24 s | 184.32 s / 3.07 min | Same shared caption and curriculum as LoRA150 |
| LoRA200 | 12 | 32 | 20.48 s | 655.36 s / 10.92 min | Per-track TXT sidecars plus trigger/instrumental boilerplate |

All use six separate source recordings for validation; none of these six
recordings participates in training. The original 150-step recipe uses paired
scored/hybrid-score-dropout views and two neutral replay examples. The 200-step
recipe has zero score dropout, one neutral replay example and separate neutral
retention evaluation. Replay, prefix variants and repeated updates do not count
as additional unique real audio.

AR200 prepared 34 candidate training windows, admitted 32 by structural score
triage, and retained all six validation windows. Thus 38 admitted real windows
represent 18 recordings, not 38 training tracks. Triage is not manual verification
of pitch, chords, timing or local key. The older models and raw manifests remain
unchanged.

## Full folder and caption inventory

The current `koan_dset` contains **42 audio files and 42 TXT sidecars**, totaling
87.49 minutes. **38 recordings are at least 60 seconds long**; four are shorter
(45.18–50.23 seconds). There are no byte-identical source duplicates. Twelve
recordings trained, six were held out, and 24 have not entered these AR runs.
The training/validation subset consists of the `koan_sound_*` files; much of the
rest of the curated folder's stylistic variety has therefore not been used.

The complete folder has 12 distinct genre-description strings after separating
tempo/key metadata, including neurofunk/drum-and-bass, breakcore, synthwave,
ambient electronica and funk. AR200 has 12 distinct full training captions,
but their genre portion is the same: `glitch hop, neurofunk, electronica,
intricate breakbeat`. Their differences are primarily BPM/key. Rich per-window
instrumentation/production captioning has not been implemented.

The single UI **use Koan prompt** button is a convenience inference prompt. It
does not imply that the AR200 training captions were all replaced by that text.
AR200 actually uses, for example:

```
koan_palette. Instrumental music. glitch hop, neurofunk, electronica,
intricate breakbeat, 130 bpm, E minor.
```

The audit verifies the source hashes, current sidecar text, used-score hashes
and prepared-prefix hashes against the admitted-window manifest.

## Tempo/key consistency needs work

The current AR200 recipe contains BPM/key **both in captions and in the ABC
score**, but the ABC comes from separate excerpt transcriptions. Its estimated
headers were not systematically reconciled with the source sidecars.

Among 32 admitted training windows, tempo is exact in 15, within 1.5 BPM in 12,
and approximately half/double-time in five. Key is exact in 14, relative or
enharmonic in three, and has a different key signature in 15. Validation has
three differing key signatures and one substantial tempo discrepancy. For
example, Drift's caption says 86 BPM/G minor while some used excerpts say
171 BPM/C minor; one Aeon validation excerpt says 43 BPM against the sidecar's
114 BPM. These are review flags, not proof the sidecar or the local transcription
is always wrong: local modulation, relative keys, beat interpretation and
estimator errors all need to be distinguished.

The next dataset should give the ABC score the primary tempo/key role and use
`koan_palette` plus the sidecar's actual genre/production description as the
style caption. For the user's example:

```
caption: koan_palette. Instrumental music. glitch hop, neurofunk,
         electronica, intricate breakbeat.
score:   Q:1/4=88
         K:Dm
```

The score must encode the intended pitches and audio-relative note times.
Changing `Q:` alone changes playback speed; changing `K:` alone can change
implicit accidentals and therefore pitches. Re-engrave from absolute note/event
times and pitches under a reviewed beat grid/key rather than patching headers.
Do not claim that renaming a key fixes incorrect transcription notes. Accurate
duplicate BPM/key in captions is valid too, but contradictory duplication should
not be the default recipe. This recommendation is an experiment policy, not
evidence that stripping metadata alone improves quality.

## What the toolkit's 60 seconds means

The current [toolkit source](https://github.com/ostris/ai-toolkit/blob/main/extensions_built_in/audio_models/yue2/yue2_model.py)
defaults `train_window_frames` to 1,500 at 25 Hz: a **NAR acoustic crop** of up
to 60 seconds. Its auxiliary AR loss separately uses the song from its start,
with `ar_max_tokens=0` meaning the whole cached song by default. The 60-second
setting is not a complete description of its AR training distribution. The
current source was fetched and archived as text; remote code was not executed.

Our AR currently has a restricted codec/MUSIC_END vocabulary, semantic-only
loss positions, a maximum of 2,048 target frames and a 4,096-position training
graph cap. Its CLI requires equal target-frame counts in a run. A 1,500-frame
configuration is within the nominal frame cap, but memory feasibility is not
established, especially with longer score prefixes. The successful 512-frame,
618-prefix rank-32 probe used about 6.3 GiB on the 8GB GPU. Native manual
attention and readout/KL workspaces still require a resource profile for longer
sequences. Shorter recordings also require real length buckets or correct
masking; silent padding or fake repeated content is not an acceptable substitute.

## Bounded remaining experiment

1. Prepare the full curated folder with verified source/sidecar identities,
   preserve the six current whole-recording holdouts (36 training / six
   validation recordings), and report coverage per recording/genre. Include the
   four shorter recordings naturally. Use varied, aligned musical passages
   rather than repeatedly visiting the same fixed excerpt. Check transcription
   quality and caption/score consistency before fitting.
2. Profile 30-second and 60-second AR contexts and implement the needed length
   buckets/masking. Use the longest practical context with correctly aligned
   scores. Add genuine end supervision only to confirmed musical endings;
   neither a crop nor an arbitrary source-file boundary proves a musical ending.
   Measure any GGML/operator requirement before changing the submodule.
3. Run matched rank-32 LoRA and DoRA on that same expanded scored-only recipe,
   with common-model planning, the same caption policy, KL/replay, optimizer and
   semantic-token budget. Do not infer a meaningful training budget from
   “200 versus 950 updates” when context/coverage differ. Select by heldout
   metrics plus fixed-score/fixed-seed palette, content and fidelity auditions.
4. Calibrate NAR only for the selected AR candidate. Prefer one broadly useful
   default and retain the current three useful pairs as alternatives. Do not
   add direct ABC/full-vocabulary planning loss to this comparison: it is a
   separate optional objective experiment if score following remains inadequate.

This audit does **not** launch a new fit or modify dataset files, adapters, the
UI or the running servers. It resolves what has been trained and defines the
next resource/data gate before another optimization run.

[Portable inventory and window/header audit](experiments/yue2-koan-ar-data-audit-2026-10-26.json).
Raw results, the audit script and archived upstream text are under
`outputs/koan-ar-expansion-audit-2026-10-26/` (the script is
`outputs/audit_koan_ar_data.py`).
