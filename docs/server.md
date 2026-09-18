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
# --instrumental-lora REF[=SCALE]
# --continuation-lora REF[=SCALE]
```

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
prefix, releases it, and only then loads generation. Send
`"keep_models": true`, or start with `--keep-models`, to stay resident, and
`POST /unload` to release.

Supervisors running Yuey on a shared GPU can start with `--force-unload`. It
overrides request-level `keep_models:true`, ensuring each success, failure, or
cancellation releases the transcriber and generator before the job finishes.

## Routes

| Method | Path | Result |
|---|---|---|
| `GET` | `/` | Embedded standalone Yuey UI |
| `GET` | `/health` | `{status, service, version, device, encoding, busy, queued, generation:{available, loaded, model, vae, tokenizer}, transcription:{available, loaded, model}}` |
| `GET` | `/props` | Local device/VRAM inventory, installed model tiers, recommendation, defaults, and UI capabilities |
| `GET` | `/loras` | `{success, adapters_dir, loras:[{index, name, path}]}` for `*-LoRA.gguf` files |
| `POST` | `/plan` | text to an editable ABC plan without semantic/flow/VAE work |
| `POST` | `/generate` | text, or text plus ABC, to a song → `{success, session_id, seed, status}` |
| `POST` | `/cover` | `audio_data` → transcription → song → `{success, session_id, seed, status}` |
| `POST` | `/continue` | `audio_data` → score + semantic prefix → extended song |
| `POST` | `/transcribe` | `audio_data` → ABC, MIDI, events → `{success, session_id, status}` |
| `GET` | `/poll_status/<id>` | progress; results on completion; `?consume=1` removes a finished job |
| `POST` | `/cancel/<id>` | cooperative cancel; a queued job fails immediately as `cancelled` |
| `POST` | `/unload` | release resident models; `409` while a job runs |

Errors are `{success:false, error}` with an HTTP status. Unknown sessions
return `404`, as in sa3-server.

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
  sections in the score's section order, and replaces Vocal notes with
  duration-equivalent rests before semantic generation. For a newly planned
  song it seeds a conventional intro/verse/chorus form, then rebuilds the
  sections from the completed score.
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
  only and requires `bpm` plus a major/minor `key`; meter defaults to 4/4. The
  server validates the values and constructs the proven two-voice planning
  prefix before sampling. It is mutually exclusive with `abc` and
  `abc_prefix`. Omit the object entirely for automatic musical planning.
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

- **`audio_format`**: `wav` is 16-bit PCM, which gary4juce reads; `wav_float`
  keeps the model's float output.

`/cover` also requires `audio_data`, a base64 WAV of any rate and channel count.
It accepts `transcription_mode`: `melody` (default) or `full`. A `full`
transcription conditions `symbolic_mode: full` unless overridden. `/cover`
rejects `abc`; the score comes from the audio. Set `instrumental: true` for an
instrumental-source remix: the server rests SheetSage2's Vocal lane and uses
the same empty lyric-section/no-vocal conditioning as instrumental generation.
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
result and free it at once; a song's base64 WAV is tens of megabytes.

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
