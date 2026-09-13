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
