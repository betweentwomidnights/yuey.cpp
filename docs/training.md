# Local style training (draft)

Open `/training` in the yuey web UI. A named run prepares and reviews a dataset,
learns the AR style adapter, freezes its validation-selected checkpoint, prepares
matched NAR conditioning, and calibrates the NAR adapter. Publication combines
the pair into one style entry with a caption pool. Planning remains base/common.

This is an instrumental training workflow. **All model computation uses native
GGML**, including VAE posterior-mean encoding. Python coordinates manifests,
provenance and processes; the training add-on does not require or import PyTorch.
The optional PyTorch encoder remains a developer-only numerical oracle.

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
VAE encoding follows `device` by default. Optional `vae_device`, `vae_binary`
and `vae_core_frames` settings select an independent installed backend/tool and
tile profile. CUDA and Vulkan encoding have been checked on the laptop; Metal
uses the same GGML graph but awaits testing on Apple hardware. See the
[native encoder validation](training-history.md#native-vae-encoder-2026-10-27).

The tested machine has an RTX 5070 Laptop GPU with 8 GB VRAM and **64 GB host
RAM**. Recipes 1–3 hold every AR teacher distribution in host RAM (about
15.5 GiB working set and 22.5 GiB private on the 157-window Koan corpus).
Recipe 4 streams them from disk instead: about 3.1 GiB working set on the same
corpus, independent of the number of windows, plus about 94 MiB of disk per
30-second training window. Dataset review shows both estimates. Full-corpus training
diagnostics can dominate a run; new UI jobs validate all heldout recordings and
any retention examples by default. CUDA is the measured GPU training backend.
Use one inference server per device; unrelated servers/applications are outside
this server's device reservation.

Unload models in other applications before starting training on an 8 GB card.
On Windows, a trainer started under GPU memory pressure can stay slow even
after the other model unloads. In the local 30-second LoRA test, updates took
about 86 seconds with 170 MiB reported as shared GPU memory; a fresh trainer
after unloading the competing model used the normal 74 MiB shared allocation
and returned to 3.6-second updates. Use **Stop**, wait for its checkpoint, then
**Resume** after freeing GPU memory. See the [recovery measurements](training-history.md#windows-gpu-residency-slowdown-2026-10-27).
AR progress distinguishes teacher preparation, initial/checkpoint evaluation
and updates, and reports the most recent update duration. Startup and full
evaluation time are additional to that update duration.
Both expert charts show completed evaluations during fitting, using the same
recording-balanced heldout loss as final checkpoint selection. “Best so far”
remains provisional until the expert finishes. The update counter reaching its
ceiling (for example 800/800) still leaves final validation and the checkpoint
save; AR progress identifies that phase and shows its elapsed time. A full
evaluation can take several minutes on this laptop.
NAR startup also validates every frozen conditioning cache on the CPU before
loading the GPU model. Its initial evaluation runs the configured validation
windows at three fixed noise levels before update 1. Native progress reports cache files
validated and evaluation windows completed separately from optimizer updates;
0/400 during these phases does not mean the worker is idle. Validation and
initial evaluation run again when the first fit starts after preflight. A NAR
resume revalidates the caches and restores optimizer state, but skips the initial
baseline evaluation. Existing runs retain their frozen native executable; older
executables expose activity/elapsed time, while new runs provide exact counters.
New requests are pinned to recipe version 4. It keeps everything in recipe 3
and adds a streamed AR teacher cache. Before the first fit, the native trainer
writes one F32 GGUF per training window (`.partial`, then renamed), and the
coordinator records each file's SHA-256 in `teacher/index.json`. The fitter checks
every header and exact prefix/codes at startup, decodes one window per update,
and never recomputes teachers on resume. The coordinator re-hashes the cache
before and after every fit. See [the teacher cache measurements](training-history.md#recipe-4-streamed-ar-teacher-cache-2026-10-28).
Recipe 3 retains recipe 2's heldout
validation and maximum-context NAR preflight. Preflight verifies cache
checksums and inspects all headers, then tests one training sample padded to the
maximum context across training, holdout and retention caches. The fitter still
validates and fingerprints the complete dataset before its first update. Recipe
3 fingerprints encoded files without expanding every cache to F32 first, while
retaining all coordinator SHA-256 checks and native provenance/shape checks.
Recipes 1 and 2 retain their original policies and frozen executables. Fingerprint
versions are bound to checkpoints; switching versions requires a new run.
See [handoff measurements and parity checks](training-history.md#recipes-2-and-3-preflight-and-fingerprints-2026-10-27).
The [final optimization measurements](training-history.md#recipes-2-and-3-preflight-and-fingerprints-2026-10-27)
cover the byte audit and real-window CUDA parity.

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
| Save / heldout validation | Every 100 updates and the final update |
| Validation scope | All heldout recordings and any retention examples; full training diagnostics optional |
| Selection | Minimum recording-balanced heldout AR CE / NAR flow MSE |
| Sampling | Shuffled recording-balanced windows |
| Conditioning | AR-only CUDA prefill, original precision, lossless F16 K/V, resident model reuse |
| VAE targets | Native GGML F32 encoder, 128-frame cores, 16-frame internal halos; native audio reader/polyphase resampler |

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
PID-reuse mistakes. The worker waits for committed ownership before touching
the device, recognizing the Windows venv redirector as well as a direct Python
process. A newly constructed server adopts an existing worker from the
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

## Disk use

Most of a run's disk is two caches derived from the frozen AR expert, not the
dataset. On the 166-window, 38-recording laptop corpus (30-second windows):

| Item | Size | Lifetime |
| --- | ---: | --- |
| NAR conditioning (AR K/V, F16, 112 KiB per token) | 26.1 GiB | Until publish |
| AR KL teachers (recipe 4, F32) | 12.6 GiB | Until AR selection |
| Checkpoints, rank 32 (adapter plus AdamW state) | about 0.56 GB each | Step zero, heldout-best and latest kept |

Fits keep only step zero, the heldout-best and the newest resumable checkpoint
per expert. A checkpoint is removed only after its evaluation is recorded and a
newer complete state exists, so selection and resume are unaffected. The AR
teacher cache is released once AR selection completes, before NAR conditioning
is written, so peak disk is the larger of the two caches rather than their sum.
Dataset review reports `estimated_peak_disk_gib` and `free_disk_gib`; training
refuses to start when the remaining caches and checkpoints would not fit.

Publishing then releases the conditioning caches, optimizer states, other
checkpoints, probe diagnostics and audio slices, keeping the selected adapters,
manifests, metrics and logs. A released run cannot train again; start a new
run. Set `"keep_caches": true` when creating a run to keep everything for later
recalibration.

## HTTP surface

- `GET /training`: embedded trainer.
- `GET /training/api/runs`: configured status, durable run history and progress.
- `POST /training/api/runs`: create a unique run and prepare its dataset.
- `POST /training/api/runs/<id>/train`: train the matched AR/NAR pair.
- `POST /training/api/runs/<id>/stop`: request cooperative checkpoint/stop.
- `POST /training/api/runs/<id>/resume`: continue verified preparation or fitting.
- `POST /training/api/runs/<id>/publish`: register the reviewed pair, then
  release training caches unless the run set `keep_caches`.

See [conditioning benchmarks](training-history.md#conditioning-cache-optimization-2026-10-27) for measured
speed/memory evidence and numerical limits. SA3 and gary4local remain unchanged;
the named-run/prompt-pool contract is intended to support their later integration.
