# Generation listening tests

Listening fixtures complement numerical parity tests. They are intentionally
long enough to establish a phrase and exercise both the planning and synthesis
branches.

For each scratch request, render two comparisons:

1. **End to end:** every model tier independently plans ABC and synthesizes
   audio. This exposes quantization changes in planning as well as rendering.
2. **Fixed plan:** reuse the BF16-generated ABC with every quantized tier. This
   isolates semantic generation, flow, and decoding from planning divergence.

Use the same seed, sampling settings, semantic length, ODE step count, and F16
VAE across tiers. Autoregressive renders will diverge and should not be judged
sample-for-sample. Listen for musical structure, prompt adherence, melody and
lyric alignment, vocal stability, transients, stereo image, texture, and noise
floor. Note abrupt endings caused by a forced semantic-frame limit separately
from interior quality.

The transcription model is not part of this quantization matrix. Its F16 GGUF
is already small enough for the target consumer hardware and remains the
accuracy-preserving default.

## Locked musical structure

Free-form tags are soft conditioning: even the released BF16 planner may choose
a different key, tempo, or meter. The native request therefore accepts an
optional `abc_prefix`, placed exactly after the model's `ABC_START` token and
before symbolic sampling. A host can supply a validated header containing its
locked meter, unit length, tempo, voices, and key; YuE2 then composes the score
body under that prefix. `abc_prefix` is mutually exclusive with a complete
external `abc` score and must end with a newline.

The prefix is the low-level transport primitive, not the intended UI. A DAW or
server should construct it from typed key, BPM, and meter fields, then parse and
validate the completed plan before semantic generation begins. Changing only a
completed score's `K:` or `M:` field is unsafe because the notes, chords, rests,
ties, and bar lengths would no longer agree with the header.

The first input-audio cover and score-continuation experiment is recorded in
[sa3-8bar-e-major-results.md](sa3-8bar-e-major-results.md). It demonstrates that
a duration request is a hard semantic ceiling rather than ending conditioning:
the same completed plan ends cleanly when allowed to emit `MUSIC_END` and cuts
active audio when capped early.
