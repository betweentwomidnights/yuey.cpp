# changelog

## v0.3.0

**train your own style.** `/training` in the web UI takes a folder of
recordings with TXT captions (style, tempo and key), reviews the dataset, fits
the AR style adapter, freezes its heldout-best checkpoint, prepares matched
conditioning and calibrates the NAR adapter. publishing puts the pair in the
models folder as one style with a caption pool, and it shows up in `/loras`
for inference. runs are named and durable: they survive server restarts, stop
after the current update and resume where they stopped. see
[docs/training.md](docs/training.md).

**training is native.** `yue2-train-run` runs every stage, and `yue2-server`
launches it for each run, so training needs no Python and no PyTorch:
`yue2-train-run setup --bin BIN --models-dir MODELS --out training-runtime.json`
writes the config `--training-config` reads. it replaced the Python workflow
one stage at a time, each checked equal to the Python before the Python was
removed, so runs the Python started still resume. `yue2-train-run
capabilities` reports the recipes, request settings and storage modes an
installation supports.

**it trains on the Q4_K_M base.** an installation with only the inference
model can train; `setup` uses BF16 when it's installed and Q4_K_M otherwise.
on an 8 GB RTX 5070 Laptop at 30-second windows and rank 32, Q4_K_M trains at
BF16's speed with 2.7 GiB less VRAM for AR and 1.8 GiB less for NAR. its
gradients match a dequantized reference (cosine 0.9999997). both bases carry
the same checkpoint fingerprint, so a style trained on either loads on either.
the base itself differs, so a style trained on Q4_K_M can sound different from
one trained on BF16.

**storage modes trade disk for time or VRAM.** recipe 5, the default, adds
`teacher_storage` (`cache`, `online`) and `conditioning_storage` (`cache`,
`rolling`, `resident`) to a run request; every mode gives the same results as
the cached default. dataset review estimates each mode's disk and VRAM, and
finished fits prune their checkpoints and release their caches after
publishing (`keep_caches` keeps them).

**the semantic tokenizer reads FLAC.** `yue2-semantic-tokenize` read WAV only,
so a FLAC dataset failed at its first semantic cache. a WAV decodes exactly as
before.

**ggml moves from `4ad3b30b` to `b77ae790`,** the shared fork on upstream GGML
0.27. it keeps the training kernels, backward ops and quantized `OUT_PROD`
(restored on Vulkan, which the merge had dropped) and carries the Metal Q4_K
scale precision fix. training results were byte-identical across the move.

**packages.** the Windows core zip carries `yue2-train-run` and the native
trainers beside the server, so a supervisor gets training with the runtime it
already installs; the standalone zip no longer ships Python scripts. local
training ownership runs on Windows and Linux; the macOS package has no
training.

## v0.2.4

**the Visual C++ runtime comes with the package.** the core zip now carries
`msvcp140`, `vcruntime140`, `vcruntime140_1` and `vcomp140` beside the
executables, from the Visual Studio that built them. Windows loads those before
System32, so yuey runs on a machine with an old or missing Visual C++
Redistributable, and a supervisor never needs an administrator install of one.
the package script fails if the runtime is older than the compiler, and
`BUILD-INFO.json` records its version. nothing else changed since v0.2.3.

## v0.2.3

**generation progress follows the work.** `/poll_status` used to reach 80%
before flow started, though flow is most of a generation's time: an M4 Air
render of 32 seconds of audio took about 15 seconds of semantic generation and
75 seconds of flow. progress now gives ABC planning 0–2%, semantic tokens
2–20%, flow 20–97% and decode 97–99%. a cover still reserves the first 15% for
transcription and a real-audio continuation the first 25% for its prefix.
progress never goes backwards, an unknown total leaves a stage at its start,
and only a finished result reports 100%. it's a work estimate, not an ETA.

**ggml moves from `60f49e09` to `4ad3b30b`.** that brings a tiled Metal im2col
kernel for batch-1 convolutions, which the VAE decode now uses, and a CPU
`im2col_f32` that splits work over output pixels. both only move data. the
Vulkan change keeps F32 kernels in fp32 for direct 2D/3D convolutions, which
yuey doesn't call. ggml-bump-check found every output identical on CUDA,
Vulkan, CPU and Metal.

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
