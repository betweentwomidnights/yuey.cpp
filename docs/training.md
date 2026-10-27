# Local style training (draft)

Open `/training` in the yuey web UI. A named run prepares and reviews a dataset,
learns the AR style adapter, freezes its validation-selected checkpoint, prepares
matched NAR conditioning, and calibrates the NAR adapter. Publication combines
the pair into one style entry with a caption pool. Planning remains base/common.

This is an instrumental training workflow. It uses native GGML optimization and
Python preprocessing, including the current **CPU VAE encoder**. The CUDA
conditioning optimization does not move that encoder to the GPU.

## Setup

Use 64-bit Python with the packages in `tools/requirements-training.txt`. For a
source checkout, create a separate virtual environment and build the normal
tools. A standalone Windows package carries these tools and scripts, but does
not bundle Python or download models. Inference works without the training add-on.

```powershell
python -m venv .venv-training
.venv-training/Scripts/python -m pip install -r tools/requirements-training.txt
.venv-training/Scripts/python tools/setup_training.py --bin build/bin/Release --models-dir models
build/bin/Release/yue2-server --models-dir models --training-config training-runtime.json
```

For an unpacked standalone package, use `--bin .`. Setup requires the BF16
generation model, folded F16 VAE, v9 semantic tokenizer, SheetSage2 transcriber,
Qwen text tokenizer and instrumental companion. `training-runtime.json` records
absolute paths, Python, device, native binary folder, managed run root and model
publication folder. Use that same models folder as the inference adapter folder.
Model/runtime paths are installation settings; the browser cannot override them.

The tested machine has an RTX 5070 Laptop GPU with 8 GB VRAM and **64 GB host
RAM**. The AR teacher distribution cache uses substantial host memory; an 8 GB
GPU fit does not establish a small host-memory requirement. Validation is full
and can dominate a small training run. CUDA is the measured GPU training backend.
Use one inference server per device; unrelated servers/applications are outside
this server's device reservation.

## Dataset and recipe

Provide at least two distinct WAV/FLAC recordings, with matching UTF-8 TXT
sidecars. Captions currently require explicit BPM and major/minor key, e.g.
`glitch hop, neurofunk, intricate breakbeat, 88 bpm, D minor`.
The style words and trigger become the caption; tempo/key enter the normalized
score. Source audio and captions stay read-only. Duplicate audio is rejected.
Scores retain absolute transcribed notes and triplets on a 12-tick quarter grid.

Preparation uses full-track v9 semantic encoding and real, bar-aligned windows
throughout every recording. Roughly 15% of recordings (at least one) become
deterministic holdouts. Neighboring windows never straddle training/validation.
Shorter-than-window recordings and missing metadata require dataset correction;
they are not padded or silently assigned invented captions.

Review recording splits, captions, window coverage, exclusions and estimated
conditioning disk use, then start **Train AR + NAR**. The advanced defaults are:

| Setting | Default |
| --- | --- |
| Window | 30 seconds |
| AR | DoRA, rank 32, LR 2e-5, teacher KL 0.2, ceiling 800 updates |
| NAR | LoRA, rank 32, LR 3e-5, uniform noise schedule, 400 updates |
| Save / full evaluation | Every 100 updates and the final update |
| Selection | Minimum recording-balanced heldout AR CE / NAR flow MSE |
| Sampling | Shuffled recording-balanced windows |
| Conditioning | AR-only CUDA prefill, original precision, lossless F16 K/V, resident model reuse |

These defaults follow the full Koan experiment, whose selected pair was AR
DoRA500/NAR400. They do not force step 500 for a new dataset. **The initial UI
workflow does not generate the neutral replay/retention examples used in that
research run**; generalization and musical defaults still need broader listening
validation before release. The underlying research CLI continues to support
explicit auxiliary examples. Rank/family, learning rates, budgets, seed, KL and
evaluation cadence are advanced controls. NAR uses LoRA; planning training and
score dropout are not product controls.

60-second windows are experimental: full AR training spilled into WDDM shared
memory on this laptop and was much slower. Faster cache preparation does not
change that training limit.

## Run ownership, stopping and publication

Run IDs/names are unique. Requests and resolved configuration remain immutable;
helpers verify model, executable/backend, source and cache hashes. Stop requests
checkpoint after the current native update, or finish the current preprocessing
item. A stop request retains ownership until the worker exits. Resume uses the
native optimizer, noise RNG and recording/window sampler state in a new segment.
Completed-fit selection can be recovered without adding updates.

The server unloads its inference models before launching the worker and rejects
new compute requests while the worker lives. PID plus creation identity prevents
PID-reuse mistakes. A newly constructed server adopts an existing worker from the
registry. On Windows, a worker-owned Job Object terminates descendants if the
coordinator crashes. Normal monitoring errors still wait for the child. Worker
status and checkpoints survive closing the browser; keep the local runtime and
dataset paths available. Interrupted/incomplete preprocessing with missing
provenance fails visibly rather than adopting unverified bytes.

After both stages, review the separate metrics and audition checkpoints. Publish
explicitly when satisfied. Publication verifies that all combined expert tensors
match the selected source checkpoints, retains DoRA normalization metadata, and
writes `<run-id>-F32-LoRA.gguf` plus `.training.json`. The sidecar records source
hashes, base compatibility, required companion order/strength, trigger and real
training captions. The inference dropdown uses the display name and its caption
button draws from that pool; old named choices/preferences are preserved.

The API is local, like the inference server. Do not expose it to untrusted
clients: dataset folder paths refer to the server's filesystem.

## HTTP surface

- `GET /training`: embedded trainer.
- `GET /training/api/runs`: configured status, durable run history and progress.
- `POST /training/api/runs`: create a unique run and prepare its dataset.
- `POST /training/api/runs/<id>/train`: train the matched AR/NAR pair.
- `POST /training/api/runs/<id>/stop`: request cooperative checkpoint/stop.
- `POST /training/api/runs/<id>/resume`: continue verified preparation or fitting.
- `POST /training/api/runs/<id>/publish`: register the reviewed pair.

See [conditioning benchmarks](training-optimization-2026-10-27.md) for measured
speed/memory evidence and numerical limits. SA3 and gary4local remain unchanged;
the named-run/prompt-pool contract is intended to support their later integration.
