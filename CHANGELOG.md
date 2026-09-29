# Changelog

## v0.2.0

The first release with prebuilt Windows packages: split core and GPU-backend
zips for gary4local, and a standalone zip with everything, the CUDA runtime
included, for running yuey on its own. See [packaging](docs/packaging.md).

- **Server for gary4juce.** `yue2-server` speaks the gary4local job contract
  (`/generate`, `/cover`, `/continue`, `/transcribe`, `/plan`, polling and
  cancel) on port 8007, with `--props` and `--version` for installers.
- **Speed.** On CUDA, the autoregressive decode step replays as a CUDA graph,
  roughly halving decode time.
- **Vulkan.** SheetSage2 transcription, and so covers, remixes and
  continuation, now works on Vulkan. Its decoder used to drift into one
  repeated note after a bar or two. This matters most for AMD and Intel GPUs,
  where Vulkan is the only backend.
- **Instrumental.** Instrumental requests move the sung or planned melody to
  the instrument lane, as official YuE does, instead of resting it
  (`instrumental_method`, default `transfer`; `rest` keeps the old behavior).
  `YUE2_INSTRUMENTAL_METHOD` and `YUE2_USE_INSTRUMENTAL_ADAPTER` set server
  defaults.
- **Remixes from transcription** no longer come out at half time, and no
  longer open with a bar of silence where the tracker's first downbeat came
  late.
- **Planning.**
  - The planner can choose the key.
  - Planning keys are written the way the model's scores spell them (`Eb`,
    not `D#`).
  - `natural_max_seconds` caps planner-chosen length.
- **Score transforms.** `POST /score/transform` applies quick edits for a
  score editor: half and double time, transposition, melody to instrument,
  lane swap, and dropping chords. `/health` lists them. Scores are read the
  way official YuE's tools read the dialect.
- **FLAC on the wire.** Uploads may be FLAC, and results come back as FLAC on
  request (`audio_format`), roughly halving the transfer.
- **Dice prompts.** `/prompts` serves a style prompt pool for a client's dice
  button.
