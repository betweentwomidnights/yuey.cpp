# Score-aligned generation

Date: 2026-09-14  
Native commit: `b26efb7`  
Backend: DGX Spark CUDA, Q4_K_M generation model, F16 VAE

The product path now expresses musical length in bars. A completed plan can be
fitted by retaining its opening plus real tail bars as an outro. Semantic stop
is disabled until the nominal end of that accepted score, after which the model
has a conservative safety budget in which to emit `MUSIC_END`.

## Early-stop finding

Before enforcing the score minimum, a fixed 16-bar score at 95 BPM ended after
832 semantic frames (33.28 seconds), before its 40.42-second score grid. With
the minimum enforced, the same request and seed emitted 1,028 frames and ended
naturally at 41.12 seconds.

## Chord causal pair

The pair uses identical notes, rests, sections, style, seed, model, and flow
settings. Only the 16 chord annotations differ: one cycles
`C#m / F#m7 / Amaj7 / B` from the first bar and one remains on `C#m`.

| Score | Semantic frames | Audio | Truncated | Final 0.5 s RMS / peak |
|---|---:|---:|---|---:|
| Four-chord cycle | 1,028 | 41.1187 s | no | -87.61 / -76.14 dBFS |
| Constant C-sharp minor | 1,040 | 41.5987 s | no | -43.70 / -28.98 dBFS |

The constant-harmony render reached `MUSIC_END`, but its final half-second is
not silence. Natural protocol completion and an acoustically faded tail should
therefore remain separate validation signals.

## From-scratch target-bar path

A separate text-to-music request exercised actual planner fitting with
`target_bars=16`, `ending=outro`, and `outro_bars=4`. The accepted score has 16
bars, `intro` and `outro` sections, 115 editable notes, and 16 chord positions
using `C#m`, `A`, `E`, and `B`. It emitted 1,046 semantic frames and reached
`MUSIC_END` naturally at 41.8387 seconds. Its final half-second measured
-87.53 dBFS RMS and -72.61 dBFS peak.

Instrumental generation remains best-effort and requires separate listening
for sung material or vocal samples.
