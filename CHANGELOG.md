# changelog

## v0.2.2

short supplied scores now render instead of failing with `invalid YuE2 AR
sampling configuration`. a two-bar MIDI clip at 150 BPM lasts 3.2 seconds;
its audio budget, including the decay tail, is 130 tokens. the sampler used
to keep a 200-token minimum, which was more than that budget allowed.

the minimum now fits within the score's budget while still covering every
score frame. longer scores, explicit token budgets and real-audio prefix
minimums keep their existing behavior. this applies to short MIDI renders
and other supplied scores on every backend.

validated on CUDA and Vulkan with the original melody retained exactly in
exported MIDI, and on the rebuilt remote Spark backend through gary4juce.
a real-weight regression test covers the two-bar, 150 BPM case.

## v0.2.1

**continuation is faster, mostly by doing less.** an 8-bar audio continuation
of a 25-second clip went from 46s to 31s on CUDA and from 37s to 32s on Vulkan
(RTX 5070 Laptop, Q4_K_M). none of it changes the model; the changes cut
work whose result was thrown away:

- **planning composed three times the bars it kept.** the planning margin
  (`planning_overrun`, 2×) multiplied the whole target, and a continuation's
  target includes its transcribed prefix, so a 10-bar clip continued by 8
  stopped at 36 bars. the margin now covers only the bars being added: 26.
- **CUDA planned at half Vulkan's speed on the same GPU.** each CUDA decode
  step attends over every allocated cache row (that's what lets it replay as a
  graph), and a planning session allocated its whole 9000-token budget up
  front, so every planning token attended over ~10k masked rows. the cache now
  starts small and grows in 1024-row steps, copying on the device. only CUDA
  grows; the other backends never attend past what's written, so they
  allocate up front as before.
- **lm_head computed all 184,704 logits per token** and copied them all back
  to the host, while the semantic sampler reads 32,769 of them (the codes plus
  the stop token, one contiguous run). decode now computes only the rows the
  phase can sample, about a tenth of each semantic decode step.
- **every LoRA layer scaled its output by 1.0** as a separate op. both shipped
  adapters have alpha equal to rank, so that was ~200 no-op kernels in every
  decode step and every flow evaluation, about 6% of each. it's skipped when
  the scale is one. on Vulkan that also lets the matmul and its add fuse, which
  rounds differently in the last bits (logits within 0.004 of before).
- **the flow copied the whole AR cache in every layer of every ODE
  evaluation** to append its own keys and values. the cache is now allocated
  with room for them, and the flow writes them in place. bit-identical.

**a CUDA bug the cache change surfaced.** at a KV length that's a multiple of
256, ggml-cuda decodes with its vector flash-attention kernel, and that kernel
replays wrong logits from a captured CUDA graph (top-1 disagreed with CPU on
30% of steps from the first replay). the cache is sized to stay off those
lengths. this is also what broke the 256-rounded window tried before v0.2.0,
which was put down to partial views at the time.

**every job logs a one-line timing summary** without any debug variable: time
per stage, tokens per second for planning and semantic, milliseconds per flow
step, and how much audio came out. it's meant for testers to paste from
gary4local's log.

**the docs said `/continue` ignores the natural-length ceiling.** it doesn't:
a natural continuation keeps its source bars and is held to the ceiling by
what it adds, so a 25s clip at gary4local's 180s comes back at about 205s.
the code was right; `docs/server.md` now says so.

## v0.2.0

**the first release with prebuilt Windows packages.** there are two shapes. the
split core and GPU-backend zips are for gary4local, which unpacks core plus the
one backend a machine needs and installs the CUDA runtime separately, once, for
every native service. the standalone zip is for everyone else: server, both GPU
backends and the CUDA runtime in one folder, plus `models.cmd` to fetch the
models. it runs on NVIDIA, AMD and Intel GPUs because ggml loads whichever
backend the machine can use. [packaging](docs/packaging.md) has the details.

**`yue2-server` speaks the gary4local job contract** on port 8007:
`/generate`, `/cover`, `/continue`, `/transcribe` and `/plan`, with polling and
cancel, the same shape as the other gary services. `--props` and `--version`
answer without starting the server, so an installer can check what a package
can do before it runs anything.

**decode is roughly twice as fast on CUDA.** standalone ggml builds without CUDA
graphs, and the growing KV window yuey used broke graph replay anyway. yuey now
attends over the whole cache and replays each decode step as a graph.

**transcription works on Vulkan.** before this, every transcription on Vulkan,
and so every cover, remix and continuation, came back as garbage. the first bar
or two were right, then the decoder settled on one note at the wrong tempo. the
cause was the cached decoder reading its keys and values through a strided view
of the cache, which ggml's Vulkan backend computes wrongly. copying the view
first gives Vulkan the same score as CUDA. that matters most on AMD and Intel,
where Vulkan is the only backend.

**instrumental requests move the melody to the instrument lane** instead of
resting it, the way official YuE's instrumental tools do. a Vocal note that
overlaps an Ins note wins, and chords stay on Vocal. `instrumental_method:
"rest"` brings back the old behavior, and `YUE2_INSTRUMENTAL_METHOD` and
`YUE2_USE_INSTRUMENTAL_ADAPTER` set defaults for a whole server.

**remixes from transcription stopped coming out at half time**, and stopped
opening with a bar of silence. the tempo was being converted with the first
meter's denominator, so a pickup in 4/8 halved `Q:`. separately, the tracker
often found its first downbeat late, and that lead-in became a bar of rests.
covers now start on the first downbeat.

**planning.** the planner can choose the key itself. a key it's given is written
the way the model's scores spell it: `K:Eb`, not `K:D#`, which no score uses and
the MIDI export read as C major. `natural_max_seconds` caps how long a song the
planner writes for itself.

**score transforms.** `POST /score/transform` does quick edits for a score
editor: half and double time, transposition, melody to instrument, swapping the
lanes, and dropping the chords. `/health` lists which ones a server has. scores
are read the way official YuE's tools read the dialect, so an accidental carries
to its letter in every octave, and a tie keeps its accidental across the bar.

**FLAC on the wire.** uploads can be FLAC, and results come back as FLAC when
you ask for it with `audio_format`. that roughly halves the one response that
finishes a job.

**`/prompts`** serves a pool of style prompts for a client's dice button.
