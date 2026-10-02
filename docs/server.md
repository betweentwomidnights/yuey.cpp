# yue2-server

A local HTTP service for YuE2 in the shape gary4juce already speaks. Every
compute request returns a `session_id` at once. One worker runs jobs in order,
and the client polls `/poll_status/<session_id>` for progress and, on
completion, base64 audio. This is the same contract as `sa3-server` and the
gary4local Python services. The server depends on nothing outside this
repository.

```bash
yue2-server --models-dir models --encoding Q4_K_M
# --host 127.0.0.1  --port 8007  --device cuda  --keep-models  --force-unload
# --model/--vae/--tokenizer/--transcription-model PATH  explicit files
# --semantic-tokenizer-model PATH
# --adapters-dir DIR  --lora PATH[=SCALE]  --threads N  --max-body-mb N
# --natural-max-seconds N  ceiling on planner-chosen length (default 180, 0 = off)
# --planning-overrun N     plan at most N times what is kept (default 2, 0 = off)
# --planning-loop-bars N   stop after N identical bars (default 16, 0 = off)
# --instrumental-lora REF[=SCALE]
# --continuation-lora REF[=SCALE]
# --instrumental-method M  transfer (default) or rest, for requests that omit it
# --no-instrumental-adapter  adapter off unless a request sets use_instrumental_adapter
# --vae-tile-frames N      latent frames per VAE decode window (default 512)
# --props    print the GET /props document and exit, without binding a port
# --version  print the engine version and exit
```

`--device` and `YUE2_DEVICE` accept `cpu`, `auto`, a device index, or a
case-insensitive substring of a device's name or description (as `--props`
lists them), not a backend family: on Apple silicon the GPU is `MTL0`, so
`--device mtl` selects it and `--device metal` fails with "requested accelerator
was not found". A set `YUE2_GPU` takes the same index-or-name selector and wins
over an accelerator named by `--device`, though not over `--device cpu`.

`--props` exists for installers. After unpacking a package, a supervisor runs it
once to learn which GGML backends actually initialized, what memory they
report, and which tier fits, before it ever starts the service. A CUDA backend
that cannot start (for example, a driver older than the toolkit the package was
built with) simply does not appear in `devices`.

Open `http://127.0.0.1:8007/` after launch. The responsive Yuey SPA is embedded
in the executable, has no runtime web dependencies, and talks only to the local
same-origin API. Its first slice covers creation, typed musical planning,
upload/transcription/remix, MIDI downloads, exact ABC editing and regeneration,
runtime/VRAM status, and installed quantization-tier selection.

`YUE2_MODELS_DIR`, `YUE2_ENCODING`, `YUE2_ADAPTERS_DIR`, `YUE2_PORT`,
`YUE2_DEVICE`, `YUE2_SEMANTIC_TOKENIZER_MODEL`, `YUE2_INSTRUMENTAL_LORA`, and
`YUE2_CONTINUATION_LORA` do the same as the flags, so a supervisor can stay
declarative.
`YUE2_FORCE_UNLOAD=1` prevents clients from retaining models between jobs.
`YUE2_NATURAL_MAX_SECONDS` sets the planner-length ceiling described under
[natural length](#natural-length).
`YUE2_INSTRUMENTAL_METHOD` (`transfer` or `rest`) and
`YUE2_USE_INSTRUMENTAL_ADAPTER=0` change what an instrumental request gets when
it leaves those fields out; a request that sets them still wins. Both are
reported with the other generation defaults in `/health` and `/props`.
Port 8007 is the next free port after the services gary4juce already addresses
(8000, 8002, 8003, 8005, 8006, and 8015).

## Models

Files are found by the [naming convention](distribution.md) under the models
directory and one level of subdirectories:

| Component | Match | Preference |
|---|---|---|
| Generation | `yue2-*-<Encoding>.gguf` (not `yue2-vae-`) | `--encoding`, or BF16 > F16 > Q8_0 > Q5_K_M > Q4_K_M > F32 |
| VAE | `yue2-vae-*-{F16,F32}.gguf` | F16 |
| Tokenizer | `sidecars/yue2-qwen.tiktoken` or `qwen.tiktoken` beside the model | |
| Transcription | `sheetsage2-mert2-*-{F16,F32}.gguf` | F16 |
| Real-audio tokenizer | `yue2-semantic-tokenizer-*-{F16,F32}.gguf` | F16 |

See [Instrumental and real-audio adapters](realaudio-continuation.md) for the
conversion commands, version pairing, and weight-license note.

Pass `--encoding Q4_K_M` on an 8 GB GPU: automatic selection prefers precision.
`GET /health` reports what resolved and what is loaded.

By default, models load when a job needs them and are released when it
finishes. This is sa3-server's frugal default and keeps a DAW machine's VRAM
free between requests. A cover transcribes, releases SheetSage2/MERT2, then
loads generation, so the two models never share the GPU. A continuation also
loads its separate unmodified-MERT semantic tokenizer, extracts the 25 Hz audio
prefix, releases it, and only then loads generation. Within a generation, a
frugal job also frees the generator once flow synthesis is done, before the VAE
loads, so the decode does not share the card with it either. Send
`"keep_models": true`, or start with `--keep-models`, to stay resident, and
`POST /unload` to release.

The VAE decodes in windows of `--vae-tile-frames` latent frames (about 20 s of
audio at the default 512) with a 16-frame halo, and its buffer grows with the
window: about 3.2 GB at the upstream 1024, 1.7 GB at 512. On an 8 GB RTX 5070
Laptop GPU, a 32-bar Q4_K_M render peaked at 7.1 GB with 1024-frame windows
and the generator still loaded, which is what ran out of memory beside a DAW;
it peaks at 3.3 GB with both changes. The window size changes the output only
by accumulation order (RMS difference about 58 dB below the signal, spread
through the song rather than at seams).

Supervisors running Yuey on a shared GPU can start with `--force-unload`. It
overrides request-level `keep_models:true`, ensuring each success, failure, or
cancellation releases the transcriber and generator before the job finishes.

## Routes

| Method | Path | Result |
|---|---|---|
| `GET` | `/` | Embedded standalone Yuey UI |
| `GET` | `/health` | `{status, service, version, device, encoding, busy, queued, generation:{available, loaded, model, vae, tokenizer}, transcription:{available, loaded, model}}` |
| `GET` | `/props` | Local device/VRAM inventory, installed model tiers, recommendation, defaults, and UI capabilities |
| `GET` | `/prompts` | `{dice:{instrumental:[...], vocal:[...]}}` style prompts for a client's dice button |
| `GET` | `/loras` | `{success, adapters_dir, loras:[{index, name, path}]}` for `*-LoRA.gguf` files |
| `POST` | `/plan` | text to an editable ABC plan without semantic/flow/VAE work |
| `POST` | `/generate` | text, or text plus ABC, to a song → `{success, session_id, seed, status}` |
| `POST` | `/cover` | `audio_data` → transcription → song → `{success, session_id, seed, status}` |
| `POST` | `/continue` | `audio_data` → score + semantic prefix → extended song |
| `POST` | `/transcribe` | `audio_data` → ABC, MIDI, events → `{success, session_id, status}` |
| `GET` | `/poll_status/<id>` | progress; results on completion; `?consume=1` removes a finished job |
| `POST` | `/cancel/<id>` | cooperative cancel; a queued job fails immediately as `cancelled` |
| `POST` | `/unload` | release resident models; `409` while a job runs |
| `POST` | `/score/transform` | `{abc, op}` → the edited score at once; see [score transforms](#score-transforms) |

Errors are `{success:false, error}` with an HTTP status. Unknown sessions
return `404`, as in sa3-server.

### Style prompts

`GET /prompts` returns a curated pool for a client's dice button, in the shape
the other services in this family already serve:

```json
{"dice": {"instrumental": ["..."], "vocal": ["..."]}}
```

It is static, so it answers before any model is loaded and costs nothing to
call. Two buckets, because there is one toggle that matters: an instrumental
job has `Instrumental, no vocals, no singing, no humming.` prefixed to its
style, so those prompts never mention a voice.

The prompts follow the order YuE2's own guidance sets out -- genre and
subgenre, mood and energy, lead vocal type where there is one, the instruments
as an actual band, then how it moves -- with two deliberate omissions.

Dance music has the largest share of both buckets, since most of the people
rolling are producers. A vocal-bucket prompt names a singer only where the
voice is the point: with the instrumental toggle off yuey sings anyway, so at
least a quarter of that bucket leaves the voice to the model.

Neither tempo nor length appears. A caller sets tempo through `planning.bpm` or
the `Q:` field of a score it supplies, and in a host that tempo is the
project's; length comes from `target_bars` or the natural ceiling. A prompt
claiming either can only disagree with the thing that actually decides it, and
contradictory tags are what the upstream guidance warns degrades output.
`tests/server_support_test.cpp` enforces both rules, along with no duplicates,
no vocal words in the instrumental bucket, and the voiceless quarter of the
vocal bucket.

### Local runtime properties

`GET /props` is the metadata-only bootstrap contract for the standalone UI. It
follows acestep.cpp's server-owned properties pattern: clients discover what is
actually present instead of duplicating model filenames or backend assumptions.
The response includes:

- every GGML device with backend, device type, stable device ID when available,
  and current free/total memory;
- a conservative local recommendation across BF16, Q8_0, Q5_K_M, and Q4_K_M;
- all recognized generation, VAE, transcription, tokenizer, and adapter files;
- installed state, file size, memory fit, and friendly label for every generation
  tier; and
- explicit capability flags. Structured piano-roll editing and automatic model
  downloads remain false until those server contracts exist. The first UI uses
  lossless ABC source editing and clearly labels downloads as awaiting a stable
  published manifest.

GGUF classification uses `yue2.component` and
`yue2.quantization.encoding`/`general.file_type`. A filename fallback retains
pre-v1.0 local conversions. Device memory comes from GGML's common backend API,
not CUDA-specific calls, so the same UI contract covers CUDA, Vulkan, Metal,
HIP, and future shared backends. The recommendation thresholds are deliberately
conservative working-set budgets and are separate from the displayed model-file
sizes.

## User-facing modes

The low-level routes support three distinct generative product modes. A client
should name them according to musical intent rather than expose the internal
pipeline stages directly:

1. **Create** sends text and optional structured musical controls to
   `/generate`. YuE2 plans a new ABC score and renders it.
2. **Cover** sends input audio to `/cover`. SheetSage2 transcribes the audio and
   YuE2 renders a new performance of that score. It does not extend the score.
3. **Continue** sends the source WAV to `/continue`. SheetSage2 supplies the ABC
   planning prefix while the dedicated real-audio tokenizer supplies a 25 Hz
   semantic prefix. YuE2 extends both, and the matching continuation NAR adapter
   reconstructs the source-plus-tail result. This is audio-semantic continuation
   rather than the older cover-then-continue approximation, though the result is
   still a model reconstruction rather than a bit-exact waveform splice. The
   response includes the extended ABC, transcription MIDI,
   `semantic_prefix_frames`, and `semantic_prefix_duration`.

The standalone UI exposes two continuation methods. **Score continuation** is
the default: it calls `/transcribe`, then `/generate` with the recovered ABC as
`abc_prefix`. **Audio continuation** is opt-in and calls `/continue`; it carries
the source's semantic audio forward, but reconstructs the entire result through
the real-audio decoder and can therefore lose fidelity relative to the input.
The shared piano-roll editor is mounted directly inside Create or Remix instead
of acting as a third workflow. It edits `abc`, which remains the interchange
representation sent back to generation; transcription events continue to
provide the lossless timed notes, lanes, chords, key, meter, and structure.

**Transcribe** is an independent utility rather than a fourth generation mode.
It needs only the smaller SheetSage2 model and returns ABC, Standard MIDI, and
lossless events without loading YuE2 or the VAE. This supports audio-to-MIDI
dragging even when the user does not want generated audio.

### Generation requests

`/plan`, `/generate`, and `/cover` take the same song JSON:

```json
{
  "style": "indie folk, warm female vocal, fingerpicked guitar",
  "lyrics": "[Verse]\n...\n[Chorus]\n...",
  "instrumental": false,
  "use_instrumental_adapter": true,
  "instrumental_method": "transfer",
  "experimental_vocal_rest": false,
  "encoding": "Q4_K_M",
  "planning": {"bpm": 95, "key": "C# minor", "meter_numerator": 4, "meter_denominator": 4},
  "symbolic_mode": "melody",
  "seed": -1,
  "target_bars": 16,
  "ending": "outro",
  "outro_bars": 4,
  "guidance_scale": 1.5,
  "temperature": 1.0, "top_k": 100, "top_p": 0.95, "repetition_penalty": 1.2,
  "semantic_max_tokens": 9000, "semantic_min_tokens": 200,
  "abc_max_tokens": 4096, "ode_steps": 32,
  "loras": [{"name": "my-style", "strength": 0.8}],
  "keep_models": false,
  "audio_format": "wav"
}
```

- **`style`** also accepts `caption`, `prompt`, or `tags`.
- **`instrumental`** is best-effort: occasional sung material or vocal samples
  may still occur, and completed responses include a machine-readable warning.
  It requires empty `lyrics` and symbolic mode `melody` or `full`. It adds
  explicit no-vocal style conditioning, builds empty lyric
  sections in the score's section order, and takes the melody off the Vocal
  lane before semantic generation (see `instrumental_method`). For a newly
  planned song it seeds a conventional intro/verse/chorus form, then rebuilds
  the sections from the completed score.
- **`instrumental_method`** decides what an instrumental request does with the
  Vocal lane's notes, planned or transcribed. `transfer`, the default, is
  upstream YuE2's method: the notes move into Ins, so the melody is still
  played, by an instrument. Where a moved note overlaps an Ins note the moved
  note wins and the Ins note is trimmed around it; Vocal keeps its chord
  symbols and nothing else. Bars with no Vocal notes keep their exact text.
  `rest` replaces the notes with rests and leaves Ins as it was, which is how
  instrumental worked before and usually gives a sparser piece. A score the
  transfer cannot rewrite falls back to `rest` and says so in the server log.
  It applies whether or not the instrumental adapter is in use. `/props` lists
  the accepted values under `capabilities.instrumental_methods`, and `/health`
  and `/props` report the server's default as `instrumental_method`.
- **`experimental_vocal_rest`** is a diagnostic, not an instrumental mode. It
  requires empty `lyrics` plus `symbolic_mode: melody` or `full`. After planning
  or accepting an external score, it replaces Vocal notes with
  duration-equivalent rests while retaining chord positions and preserving
  every existing Ins block exactly. It makes no claim about acoustic vocals;
  retain the control render and validate the result by listening.
- **`encoding`** selects an installed generation tier for this job (`BF16`,
  `F16`, `Q8_0`, `Q5_K_M`, `Q4_K_M`, or `F32`). Omit it or send `auto` to use
  the server default. Switching the tier safely reloads the generation model
  between jobs; the transcription model is unaffected.
- **`/plan`** stops after validated symbolic planning and returns `abc` plus
  score bars, nominal duration, seed, encoding, and truncation metadata. It
  requires `symbolic_mode: melody` or `full`. Send the accepted or edited ABC
  to `/generate` to render without sampling the plan again.
- **`abc`** applies to `/generate` only. With a score the default
  `symbolic_mode` is `melody`, as the upstream cover workflow recommends.
  Without one it is `full`, and YuE2 plans the score itself.
- **`abc_prefix`** applies to `/plan` and `/generate` and is mutually exclusive with
  `abc`. It seeds planning immediately after `ABC_START`; use a validated header
  ending in a newline to lock host-provided meter, tempo, voices, and key before
  the model composes the score body. UI clients should send structured musical
  fields to a trusted header builder rather than assemble arbitrary ABC.
- **`planning`** is that trusted typed interface. It applies to `/plan` and `/generate`
  only and requires `bpm`; meter defaults to 4/4. A major/minor `key` locks the
  key too, in either spelling (`"A# major"` and `"Bb major"` are the same); the
  header writes the spelling SheetSage2's scores use, so the model sees `K:Bb`
  and `K:C#m`, never `K:A#`. Leave it empty and the header stops after the voices, so the model
  writes its own `K:` line and opening section. In 24 test plans it wrote a
  valid key every time. The server validates the values and constructs the
  proven two-voice planning prefix before sampling. It is mutually exclusive
  with `abc` and `abc_prefix`. Omit the object entirely for automatic musical
  planning: the model then picks tempo and meter as well.
- **`natural_max_seconds`** lowers this job's natural-length ceiling. It cannot
  raise it: the server takes the smaller of the request and its own
  `--natural-max-seconds`, so a shared backend keeps its policy no matter what
  a client sends. Zero means unbounded and is honored only when the server is
  itself unbounded. See [natural length](#natural-length).
- **`seed`**: absent or negative picks a random seed, reported back.
- **`target_bars`** is the preferred musical-length control. With
  **`ending: "outro"`**, yuey retains the score opening plus `outro_bars` from
  the planner's genuine tail, producing an exact bar-length score before audio
  generation. While planning, an early `ABC_END` is suppressed until this bar
  floor is reached; the model therefore continues composing instead of failing
  merely because its first proposed ending was too short. Omit it with
  **`ending: "natural"`** to keep the complete plan.
- Semantic generation normally prevents `MUSIC_END` before the accepted
  score's nominal final bar, derives a conservative safety budget beyond it,
  and then stops naturally. `max_seconds`, legacy `duration`,
  and `semantic_max_tokens` are advanced hard-ceiling overrides; they may cut
  active audio and should not be used as ordinary musical-length controls.
  They are easy to confuse with `natural_max_seconds`, which is not the same
  thing: that one bounds the *score* before any audio is rendered, so the
  result still ends musically. Reach for `target_bars` or
  `natural_max_seconds` to set length, and for these only as a hard stop.
- **`loras`** entries name an adapter in `--adapters-dir` or give a `path`.
  Adapters are bound when the model loads, so a different set reloads it.
  Omitting `loras` uses the `--lora` defaults.
- **`--instrumental-lora`** (or `YUE2_INSTRUMENTAL_LORA`) names an adapter
  automatically stacked whenever a request sets `instrumental: true`. This is
  intended for score-first instrumental AR adapters: the server forces full
  symbolic planning and, for covers, a full SheetSage2 score. Explicit request
  adapters still compose with it. Set **`use_instrumental_adapter: false`** on
  an individual request to bypass the configured automatic adapter for an A/B
  comparison without disabling the instrumental score/lyric controls.
- **`--continuation-lora`** (or `YUE2_CONTINUATION_LORA`) names the NAR adapter
  paired with the configured real-audio tokenizer. `/continue` refuses to run
  without one, preventing real-audio tokens from being decoded with incompatible
  stock NAR weights. Explicit request adapters still compose with it.

When the standard published filenames are present in `--adapters-dir` (the
models directory by default), the server discovers both official optional
adapters automatically. Explicit flags or environment variables take priority.

- **`audio_format`**: the container of the returned audio. See
  [Audio on the wire](#audio-on-the-wire).

`/cover` also requires `audio_data`, a base64 WAV or FLAC of any rate and
channel count.
It accepts `transcription_mode`: `melody` (default) or `full`. A `full`
transcription conditions `symbolic_mode: full` unless overridden.

`pickup` (on `/cover`, `/transcribe` and `/continue`) decides what happens to
audio before the first downbeat in the song's main meter. SheetSage2's tracker
often locks on late, or labels the first second or two as a short bar in
another meter (a 7/8 bar ahead of a 4/4 song), and that lead-in became a half
bar of rests at the start of every render. `drop`, the default for `/cover`
and `/transcribe`, starts the score on that downbeat: its notes go, and its
key, chord and section label carry to bar one. `keep`, the default for
`/continue`, writes it as before, because an audio continuation's score has to
stay on the source's timeline. On a 52 s clip that opened with a 7/8 lead-in,
`drop` gave 26 bars of 4/4 lasting 52 s with the melody on the first
downbeat, where `keep` gave 27 bars, 54 s, and a bar of rests first. `/cover`
rejects `abc`; the score comes from the audio. Set `instrumental: true` for an
instrumental-source remix: the transcribed vocal melody moves to the Ins lane
(or is rested, with `instrumental_method: "rest"`), with the same empty
lyric-section/no-vocal conditioning as instrumental generation.
This flag defaults to `false` at the API boundary so vocal covers are not
silently stripped. The bundled UI exposes a separate, default-on **keep
instrumental** control for remixes; it never borrows state from the Create tab.
SheetSage2 transcribes the vocal melody but does not recognize lyric text. A
vocal cover should therefore set `instrumental: false` and provide `lyrics`;
otherwise YuE2 must invent words while following the recovered melody.

`/continue` accepts the same audio, style, lyrics, instrumental, transcription,
sampling, and adapter fields as `/cover`. It rejects caller-supplied `abc`,
`abc_prefix`, and `planning` because both prefixes must be derived from the same
audio. `continuation_bars` is the number of *new* bars (1–256); zero or omission
lets the planner choose. The server converts it to a total target only after it
knows the transcribed source length. `use_continuation_adapter` defaults to
true. Setting it false rejects `/continue`, because those semantic tokens cannot
be decoded safely with stock NAR weights; clients should use the composed score
continuation workflow instead.

### Natural length

Omitting `target_bars` lets the planner choose the form and the length. That is
the only path where nothing but the model's context window bounds the result,
and the cost is not linear in a way most callers expect: the accepted score's
duration becomes the semantic *floor*, because `MUSIC_END` is suppressed until
the score's final bar. A plan that runs long therefore forces a render that
runs long, and a score can legitimately reach the context limit at roughly
sixteen minutes of audio.

`--natural-max-seconds` (default 180, `YUE2_NATURAL_MAX_SECONDS`) holds a
planner-chosen score to a wall-clock ceiling before any audio work begins. The
bar allowance is derived from the score's own tempo and meter, so the ceiling
means the same thing at any tempo, and an over-long plan is passed through
`fit_abc_score_to_bars` rather than truncated: the opening is kept and the
planner's real tail becomes the outro, so a held song still ends.

The ceiling applies only when yuey wrote the whole score itself:

| Request | Held to the ceiling |
|---|---|
| `/plan` and `/generate` with `planning` or nothing | yes |
| `/generate` with `abc` | no, the score is the caller's |
| `/cover` | no, the length comes from the source audio |
| `/continue`, or `/generate` with a transcribed `abc_prefix` | no, holding a continuation to this could cut it shorter than the audio it extends |
| anything with `target_bars` | no, the explicit bar fit already governs |

Set it to `0` for a local install where a fifteen-minute render is the user's
own time to spend. Leave it on for anything shared: a request may lower the
ceiling with `natural_max_seconds`, but never raise it past the server's.

### Planning cost

The ceiling above is applied *after* planning, so it bounds the score rather
than the time spent writing it. These two bound the stage itself. Both are
per-request, both take a server default and an environment variable, and `0`
disables either.

`--planning-loop-bars` (default 16, `YUE2_PLANNING_LOOP_BARS`,
`planning_loop_bars`) stops once that many consecutive bars are byte-identical
in both lanes. A planner repeating itself is not composing an ending worth
waiting for, so this costs nothing that would have been kept.

`--planning-overrun` (default 2, `YUE2_PLANNING_OVERRUN`, `planning_overrun`)
stops at the first structural boundary once the plan is that many times longer
than what the fit will keep. Unlike the loop stop this is a judgement call: the
last bars of a finished plan are its composed ending, and stopping exactly at
the kept length takes whatever the planner happened to be writing instead. The
margin buys room for an ending to arrive while bounding the runaway case. On
a continuation the margin covers only the bars being added, since the
transcribed prefix is kept as it is: a 10-bar clip continued by 8 stops at
10 + 8 × 2 = 26 bars, not (10 + 8) × 2.

A stop can only land where the score parses, which is a balanced Vocal/Ins
block boundary, so both are block-granular and overshoot the threshold by up
to one block.

Measured at 16 bars, 80bpm, instrumental: 31.3s planning with both off, 17.8s
with both on, and no meaningful difference in the variety of the kept score.
On natural length the two are indistinguishable, because the plan ends on its
own first. Before the instrumental tag format was corrected these looked far
more dramatic; that was a broken input making a bound look like a fix.

### Transcription requests

```json
{"audio_data": "<base64 wav>", "mode": "melody"}
```

SheetSage2 supplies two melodic lanes (vocal and instrumental) plus symbolic
chords; it does not perform drum/bass/synth source separation. Full mode's
combined MIDI is format 1 at 960 PPQ with a conductor track, each active melody
lane on a named track, and a `Chords` track. Chord labels use the released
SheetSage2 voicings and rearticulate at downbeats. The conductor carries the
same inferred tempo and meter as ABC, key-signature changes, and section
markers. Musical subbeats—not wall-clock seconds—place notes, so the result
lands on the DAW bar grid and remains correct through pickups and meter changes.

Every completed score-bearing operation (`/plan`, `/generate`, `/cover`,
`/continue`, and `/transcribe`) keeps `midi_data` as the combined compatibility
file. It also returns `midi_files`, whose base64 values are keyed by
`transcription.mid`, `melody.mid`, `melody_vocal.mid`,
`melody_instrumental.mid`, and, in full mode, `chords.mid`.

For generation and continuation, these files are derived from the final ABC
used for the render—not merely the source transcription—so their bars and notes
match the completed result.

### Score transforms

`POST /score/transform` applies one quick, deterministic edit to a score, for a
score editor to offer as buttons before a render. It answers at once: no job,
no model, and it works while a render is running.

```json
{"abc": "X:1\n...", "op": "transpose", "semitones": 5}
```

| `op` | What it does |
|---|---|
| `half_time`, `double_time` | Halves or doubles the `Q:` tempo and leaves every note alone, so the same score plays at half or twice the speed. The result must stay within 20-400 bpm. `Q:` takes whole numbers, so half of an odd tempo rounds up, and the response's `note` says so: inside a project at the old tempo, 46 against 45.5 drifts about 1% off the grid. |
| `transpose` | Moves every note on both lanes by `semitones` (-24 to 24). Unless the shift is whole octaves, the key (header, voice and inline `K:`) and every chord symbol move too, spelled as SheetSage2 spells them: keys as the planning header writes them, chord roots in sharps. A note pushed out of the MIDI range is an error. |
| `melody_to_instrument` | The instrumental melody transfer: Vocal notes move into Ins, and Vocal keeps its chords over rests. |
| `swap_lanes` | Exchanges the two lanes' notes, so the instrument part is sung and the melody played. Chord symbols stay on Vocal. |
| `drop_chords` | Removes every chord symbol and leaves notes and rests where they were, so the model harmonises the melody itself. On a score with no melody it has little to go on: a chordless scaffold comes back as drums over one chord. |

A successful response is `{success, op, changed, bars, bpm, abc}`, plus `note`
when the edit had to round. A score the edit leaves alone comes back exactly as
it went in, with `changed:false`; otherwise only bars that change are
rewritten, and rewritten notes carry explicit accidentals. Every result is
checked to render before it is returned. A score outside the native dialect,
or an edit that cannot apply, is a `400` naming the reason, and so is a score
over 1 MB (a 900-second song is tens of KB). The edits run in linear time: a
565 KB single-section score, 17,000 bars, takes about 0.1 s for any op.

Notes are read the way upstream YuE2's ABC tools read the dialect, and the
MIDI export reads them the same way. An accidental lasts to the barline and
applies to its letter in every octave, and a tie's continuation written
without an accidental keeps the tied note's pitch, across a barline too:
`^F16-|F32` is one F#. `/health` and
`/props` list the accepted ops as `score_transforms`, so a client can show only
the buttons a server supports.

On love_is_blue's transcription, `transpose` by 5 turned `K:Am` into `K:Dm`,
and the render transcribed back as D minor over Dm, Gm and A#, the source's
Am, Dm and F moved up a fourth. `half_time` rendered 127 s where the original
rendered 65 s.

### Polling

```json
{
  "success": true, "session_id": "…",
  "generation_in_progress": true, "transform_in_progress": false,
  "status": "generating", "stage": "semantic",
  "progress": 42, "step": 3180, "total_steps": 9000,
  "queue_status": {"status": "ready", "message": "semantic"},
  "seed": 1234
}
```

- **`status`** moves through `queued → transcribing → generating → decoding →
  completed`, or ends in `failed`.
- **`stage`** is finer: `load`, `transcription`, `abc`, `semantic`, `flow`,
  `decode`.
- **`progress`** (0–100) is weighted across stages. Transcription takes the
  first 15 of a cover; for semantic generation, `total_steps` is the token
  budget, and generation normally stops before it.
- **A completed generation** adds `audio_data`, `abc` (the exact score used),
  `midi_data`, `midi_files`, and `meta:{seed, duration, score_bars,
  score_duration, sample_rate, channels, semantic_frames, semantic_budget,
  abc_truncated, semantic_truncated}`.
- **A completed plan** adds `abc`, `midi_data`, `midi_files`, and score metadata
  without rendering audio.
- **A completed transcription** adds `abc`, `midi_data` (the combined base64
  Standard MIDI), `midi_files` (combined and component MIDIs), `duration`, and
  `events` (the lossless events document).
- **Failures** add `error` and `cancelled`.

Finished jobs are kept for five minutes. Poll with `?consume=1` to take the
result and free it at once; a song's base64 WAV is tens of megabytes (see
[Audio on the wire](#audio-on-the-wire)).

Every finished job, completed or failed, also writes one line to stderr saying
where its time went:

```text
[yuey] continue completed in 30.4s on cuda | transcribe 3.5s | tokenize 0.5s | load-generator 1.0s | plan 5.2s (495 tok, 95/s) | semantic 5.6s (568 tok, 101/s) | flow 13.7s (32 steps, 429 ms/step) | decode 0.7s | 47.5s of audio, 1187 frames (619 given)
```

It needs no debug variable, so a slow report from a tester can carry it. Stages
under 50ms are left out; `frames (n given)` counts the semantic frames rendered
and how many came from the caller's audio.

## Audio on the wire

Every request and response carries audio as base64, and over a network that
blob is most of the round trip: a finished song is tens of megabytes of base64
WAV in one poll response. FLAC carries the same 16-bit samples losslessly at
roughly 40% of the size. This is the contract yuey.cpp defines for the GGML
servers, matching the SA3 service gary4beatbox already uses.

- **Uploads** (`audio_data` on `/cover`, `/continue`, `/transcribe`) may be WAV
  or FLAC. The container is read from the magic bytes (`fLaC`), never from a
  field, and a 16-bit FLAC decodes to exactly the samples its WAV would.
- **`audio_format`** on a request picks the returned container:
  - `wav` (the default, and it stays the default: a client that never asks is
    never surprised) is 16-bit PCM;
  - `wav_float` keeps the model's float output;
  - `flac` is the same 16-bit samples as `wav`, losslessly;
  - `auto` answers in the upload's container, and in `wav` for jobs with no
    upload.
- **Completed responses** name the container beside the payload as
  `audio_format`, always concrete, never `auto`.
- **`/health` and `/props`** list the accepted values as `audio_formats`. Older
  servers reject a value they do not know, so a client should ask for `flac`
  only from a server that lists it. Sniffing the returned bytes as well costs
  nothing and covers a server that ignores the field.

Encoding is yuey's own (`tools/server/flac.cpp`: fixed predictors, partitioned
Rice, stereo decorrelation, no dependency). Decoding uploads uses the vendored
dr_flac (`third_party/dr_libs`), which reads any FLAC a client produces,
including libFLAC's LPC frames.

## Calling it

```bash
sid=$(curl -s -X POST localhost:8007/generate -H 'Content-Type: application/json' \
  -d '{"style":"lo-fi soul","lyrics":"[Verse]\nslow morning","target_bars":16,"ending":"outro"}' | jq -r .session_id)
until [ "$(curl -s localhost:8007/poll_status/$sid | jq -r .status)" = completed ]; do sleep 2; done
curl -s "localhost:8007/poll_status/$sid?consume=1" | jq -r .audio_data | base64 -d > song.wav
```

## Notes

- The server binds `127.0.0.1` by default and warns when bound elsewhere:
  anyone who can reach the port can submit jobs.
- One job runs at a time, as in the sibling services: a second concurrent
  generation would not be faster, it would run out of memory.
- No gary4juce client exists yet. The routes and poll fields match what the
  plugin reads for its other services, so a YuE2 panel needs no protocol work.
