# Stable Audio 3 eight-bar cover and continuation

Date: 2026-09-13  
Native commit: `6a22ad5`  
Local renders: `outputs/listening/sa3-8bar-e-major/` (ignored listening artifacts)

This fixture uses a Stable Audio 3 instrumental exported by gary4juce. It is
20.210522 seconds of stereo 44.1 kHz PCM and has SHA-256
`D4A5D64347FD4D6F69B9BDC002B03CD263154EA5A3BBD206C78C7CE066F39027`.
The source is eight bars in 4/4 at 95 BPM. The initial listening note called it
E major; a subsequent source listen established C-sharp minor as the tonal
center.

## Full-score transcription and cover

The native F16 SheetSage2 transcription reported all three known structural
facts correctly:

| Field | Result |
|---|---|
| Meter | 4/4 |
| Tempo | 95 BPM |
| Key | C-sharp minor |
| Voices | resting vocal lane plus instrumental lane |
| Harmony | recurring `C#m - A - B` material |
| Serialized score | 448 bytes, eight bars per voice |

BF16 generation then covered that exact full score with seed `831001`, 32 flow
steps, and a 60-second semantic safety ceiling. It chose `MUSIC_END` naturally
after 860 semantic frames and 34.398667 seconds. Neither the score nor semantic
stream was truncated. The result is `cover-BF16.wav` beside its exact
`cover-BF16.abc` and request/result metadata in the ignored output directory.

## Structural continuation

Both continuation runs placed the entire 448-byte transcription after
`ABC_START` through `abc_prefix`. The completed plans preserve their respective
prefixes byte-for-byte and append new bars; this is score continuation, not
raw-waveform continuation or an audio splice.

| Run | Declared key | Planned form | Score result | Audio result |
|---|---|---|---|---|
| Alternate relative-major plan | E major | chorus, interlude, chorus, outro | 45 bars/voice, 1,586 bytes, natural end | 2,577 frames, 103.078667 s, natural end |
| Source-aligned plan | C-sharp minor | chorus then extended interlude | 49 bars/voice, 1,685 bytes, natural end | 3,033 frames, 121.318667 s, natural end |

The E-major request was created before the corrected source listen. Only
`K:C#m` was changed to `K:E` in its prefix. Those relative keys share the same
four-sharp key signature, so the existing pitch notation remained valid; the
change instead gave the planner a different tonal center for the appended
material. It is retained as a useful creative-control comparison rather than
treated as the source-ground-truth run.

The source-aligned run used the untouched transcription, the same seed
(`831002`), sampling settings, and section outline. Changing the declared tonal
intent was enough to produce a substantially different and longer continuation,
which reinforces that planning comparisons need fixed prefixes as well as fixed
random seeds.

## Ending versus duration ceiling

The first render of the completed E-major plan used a 90-second ceiling. It
stopped at exactly 2,250 semantic frames/89.998667 seconds with
`semantic_truncated=true`, even though ABC planning had completed naturally.
The same exact ABC was then rendered with a 150-second ceiling and emitted
`MUSIC_END` at 103.078667 seconds.

Tail levels make the audible distinction objective:

| Render | Semantic truncation | Last 0.5 s RMS | Last 0.5 s peak |
|---|---:|---:|---:|
| Cover | no | -87.4 dBFS | -78.3 dBFS |
| E-major continuation, 90 s cap | yes | -22.8 dBFS | -5.1 dBFS |
| E-major continuation, natural end | no | -50.8 dBFS | -36.9 dBFS |
| C-sharp-minor continuation, natural end | no | -61.0 dBFS | -45.6 dBFS |

`duration` is therefore a maximum token budget, not an instruction to compose
a cadence at that wall. For a DAW-friendly short result, the host should plan
to a target number of bars, require or construct an outro/cadence inside that
score, and give semantic generation enough headroom to reach `MUSIC_END`.
At 95 BPM in 4/4, twelve total bars are about 30.32 seconds; starting from this
eight-bar source, a roughly 30-second complete arrangement needs only four new
bars, with the final bars assigned to the ending. Exact-second delivery can be
handled after that musical boundary with a small tail or time adjustment rather
than by cutting active audio.

This experiment motivates a two-phase product API: plan/continue and validate
the score first, then render the accepted score with a computed safe semantic
ceiling. It also gives the UI separate controls for cover, score continuation,
target bars, tonal interpretation, and ending behavior.

## Transcription API and MIDI drag fixture

The same source was submitted directly to `/transcribe` with `mode: full`. The
API completed in one transcription window and returned the same 448-byte ABC,
the lossless 61-event document, and a 525-byte Standard MIDI file. There are 53
note events, all in the instrumental melody lane, spanning MIDI pitches 63-80.
The vocal lane is empty. The retained subbeat grid ends exactly at step 128,
which is eight 4/4 bars.

`transcription-full.mid` is the untouched API response. It exposes two current
limitations for DAW use:

- It is SMF format 0 with one physical track; the two possible SheetSage lanes
  are represented by MIDI channels, not separate DAW tracks. This input has only
  one active channel.
- Its tempo event is fixed at 120 BPM and its tick positions preserve elapsed
  seconds rather than the inferred 95 BPM grid. It plays for approximately the
  right wall-clock duration in a standalone player but does not land on eight
  bars when imported into a 95 BPM session.

For the Ableton drag experiment,
`transcription-full-grid-95bpm-format1.mid` is a derived comparison artifact.
It uses SMF format 1 with a conductor track (95 BPM, 4/4, C-sharp minor) and a
named Instrument Melody track. Note starts and durations come from the lossless
subbeat events, so the final note ends at tick 15,360: exactly eight bars at 480
PPQ. This file is not yet emitted by the API; it demonstrates the export shape
the DAW integration should adopt.

SheetSage2 does not infer separate drum, bass, and synthesizer tracks. Its chord
labels could support an optional generated chord lane, but that would be a
derived accompaniment representation rather than source-separated MIDI.
