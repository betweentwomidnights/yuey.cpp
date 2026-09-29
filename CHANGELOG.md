# changelog

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
