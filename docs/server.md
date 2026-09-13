# yue2-server

A local HTTP service for YuE2 in the shape gary4juce already speaks. Every
compute request returns a `session_id` at once. One worker runs jobs in order,
and the client polls `/poll_status/<session_id>` for progress and, on
completion, base64 audio. This is the same contract as `sa3-server` and the
gary4local Python services. The server depends on nothing outside this
repository.

```bash
yue2-server --models-dir models --encoding Q4_K_M
# --host 127.0.0.1  --port 8007  --device cuda  --keep-models
# --model/--vae/--tokenizer/--transcription-model PATH  explicit files
# --adapters-dir DIR  --lora PATH[=SCALE]  --threads N  --max-body-mb N
```

`YUE2_MODELS_DIR`, `YUE2_ENCODING`, `YUE2_ADAPTERS_DIR`, `YUE2_PORT`, and
`YUE2_DEVICE` do the same as the flags, so a supervisor can stay declarative.
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

Pass `--encoding Q4_K_M` on an 8 GB GPU: automatic selection prefers precision.
`GET /health` reports what resolved and what is loaded.

By default, models load when a job needs them and are released when it
finishes. This is sa3-server's frugal default and keeps a DAW machine's VRAM
free between requests. A cover transcribes, releases SheetSage2/MERT2, then
loads generation, so the two models never share the GPU. Send
`"keep_models": true`, or start with `--keep-models`, to stay resident, and
`POST /unload` to release.

## Routes

| Method | Path | Result |
|---|---|---|
| `GET` | `/health` | `{status, service, version, device, encoding, busy, queued, generation:{available, loaded, model, vae, tokenizer}, transcription:{available, loaded, model}}` |
| `GET` | `/loras` | `{success, adapters_dir, loras:[{index, name, path}]}` for `*-LoRA.gguf` files |
| `POST` | `/generate` | text, or text plus ABC, to a song → `{success, session_id, seed, status}` |
| `POST` | `/cover` | `audio_data` → transcription → song → `{success, session_id, seed, status}` |
| `POST` | `/transcribe` | `audio_data` → ABC, MIDI, events → `{success, session_id, status}` |
| `GET` | `/poll_status/<id>` | progress; results on completion; `?consume=1` removes a finished job |
| `POST` | `/cancel/<id>` | cooperative cancel; a queued job fails immediately as `cancelled` |
| `POST` | `/unload` | release resident models; `409` while a job runs |

Errors are `{success:false, error}` with an HTTP status. Unknown sessions
return `404`, as in sa3-server.

### Generation requests

`/generate` and `/cover` take JSON:

```json
{
  "style": "indie folk, warm female vocal, fingerpicked guitar",
  "lyrics": "[Verse]\n...\n[Chorus]\n...",
  "abc": "X:1\n...",
  "symbolic_mode": "melody",
  "seed": -1,
  "duration": 60,
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
- **`abc`** applies to `/generate` only. With a score the default
  `symbolic_mode` is `melody`, as the upstream cover workflow recommends.
  Without one it is `full`, and YuE2 plans the score itself.
- **`seed`**: absent or negative picks a random seed, reported back.
- **`duration`** caps the song at 25 semantic frames per second.
  `semantic_max_tokens` overrides it.
- **`loras`** entries name an adapter in `--adapters-dir` or give a `path`.
  Adapters are bound when the model loads, so a different set reloads it.
  Omitting `loras` uses the `--lora` defaults.
- **`audio_format`**: `wav` is 16-bit PCM, which gary4juce reads; `wav_float`
  keeps the model's float output.

`/cover` also requires `audio_data`, a base64 WAV of any rate and channel count.
It accepts `transcription_mode`: `melody` (default) or `full`. A `full`
transcription conditions `symbolic_mode: full` unless overridden. `/cover`
rejects `abc`; the score comes from the audio.

### Transcription requests

```json
{"audio_data": "<base64 wav>", "mode": "melody"}
```

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
- **A completed generation** adds `audio_data`, `abc` (the score used or
  planned), and `meta:{seed, duration, sample_rate, channels, semantic_frames,
  abc_truncated, semantic_truncated}`.
- **A completed transcription** adds `abc`, `midi_data` (base64 Standard MIDI),
  `duration`, and `events` (the lossless events document).
- **Failures** add `error` and `cancelled`.

Finished jobs are kept for five minutes. Poll with `?consume=1` to take the
result and free it at once; a song's base64 WAV is tens of megabytes.

## Calling it

```bash
sid=$(curl -s -X POST localhost:8007/generate -H 'Content-Type: application/json' \
  -d '{"style":"lo-fi soul","lyrics":"[Verse]\nslow morning","duration":30}' | jq -r .session_id)
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
