# Streamed AR teacher cache, 2026-10-28

Training recipe 4 moves the frozen KL teacher distributions out of the AR
fitter's RAM and into immutable, SHA-256-indexed GGUF files. On the full Koan
corpus, the CUDA fitter's peak host working set falls from **15.53 GiB to
3.09 GiB**. Peak private bytes fall from **22.50 GiB to 8.19 GiB**. A resumed
segment no longer recomputes teachers. Optimizer tensors, adapters, sampler
state, losses and heldout scores are bit-identical to the in-memory path on
CPU and CUDA. Recipes 1–3 keep their in-memory behavior.

## Why the old fitter needed so much host RAM

With `kl_weight > 0`, every training window needs the common model's full
restricted-vocabulary distribution: `targets × 32769` F32 values. That is
98.3 MB for a 30-second window (750 targets). `yue2-ar-train` computed all of
them at every process start, including resumes, and kept them resident. For the
157 Koan training windows that is 14.37 GiB. The rest of the fitter's host
footprint is small by comparison.

## Design

- **Preparation.** `yue2-ar-train ... --teacher-cache DIR --prepare-teacher-only`
  builds the trainer (with zero learned factors), computes each window's teacher
  and writes `DIR/teacher-NNNNN.gguf` via `.partial` → rename. Completed files
  are revalidated and skipped, so an interrupted or stopped preparation resumes.
  A completed file is never replaced.
- **File contents** (`src/ar_teacher_cache.h`). `general.architecture =
  yue2_ar_teacher`, format `yue2-ar-teacher-v1`, precision `f32`. The file
  records the coordinator teacher identity, base and companion SHA-256s, the
  native fixed-adapter identity, device, frames, AR slots, `predict_end`,
  window index, the manifest's prefix/codes SHA-256s and entropy. It stores the
  `probabilities` tensor `[32769, targets]` plus the exact `prefix` and `codes`.
- **Fitting.** `--teacher-cache DIR` replaces in-process preparation. A
  `TrainingSampleStore` in streaming mode checks every header, provenance field
  and exact prefix/codes at startup without reading distributions
  (`teacher_cache=i/n` progress). It then decodes exactly one window when an
  update consumes it, reading the tensor straight from its file offset.
  Evaluation never needs teachers. The trainer still revalidates finiteness,
  nonnegativity and row normalization on every use.
- **Coordinator** (`tools/train_ar_run.py`). Recipe 4 fits resolve a teacher
  identity: base, companion and dataset SHA-256s, the native runtime
  (executable and DLL hashes and the precision environment), device, AR slots,
  frames and `predict_end`. That identity is written to `teacher/identity.json`
  and is immutable. If `teacher/index.json` is missing, the coordinator runs
  native preparation (stop requests terminate it safely). It then checks that
  exactly `teacher-00000…` through the window count exist, hashes them, and
  writes the index with per-file SHA-256, bytes and per-window prefix/codes
  hashes. An existing index is re-verified before the fit, and every segment
  re-verifies all files after it. Native receives `--teacher-identity` (the
  SHA-256 of the resolved identity) and the base/companion hashes, and rejects
  any file that disagrees.
- **Recipe** (`tools/training_workflow.py`). `recipe_version` 4 adds
  `teacher_cache: f32` to the AR job; new server requests pin 4. Requests at
  1–3 resolve to the identical `ar-job.json`, and therefore the identical
  command, as before. Probe and post-probe phases are unchanged; they prepare
  only one window in memory.
- **Dataset review.** The review state now also reports
  `estimated_ar_host_ram_gib` and `estimated_teacher_disk_gib`, and the
  training page shows them next to the conditioning-cache estimate.

## Measurements

The laptop has an RTX 5070 Laptop GPU (8 GB) and 64 GB host RAM. The real Koan
corpus has 157 training and 38 validation windows of 750 frames, with a
1131-slot prefix. All runs use rank 32, LR 2e-5, KL 0.2, the instrumental
companion, `GGML_CUDA_TF32=0`, heldout evaluation every 2 updates, and a 4-update
budget. Everything runs through `train_ar_run.py` with this branch's CUDA build.
Peaks are the OS `PeakWorkingSetSize` / `PeakPagefileUsage` of the native
process, which the coordinator now records in each `*.command.json`.

| Native process | Wall s | Peak working set GiB | Peak private GiB | GPU peak MiB |
| --- | ---: | ---: | ---: | ---: |
| In-memory fit (recipe ≤3 path), incl. 157 teachers | 414 | **15.53** | **22.50** | 7150 |
| Teacher preparation pass (157 windows) | 302 | 3.09 | 8.17 | — |
| Streamed fit, fresh (recipe 4) | 181 | **3.09** | **8.19** | 7150 |
| Streamed fit, resumed at update 2 | 105 | 3.09 | 8.19 | 6900 |

The private-bytes difference (14.31 GiB) equals the teacher bytes (14.37 GiB).
A sampled timeline of a streamed fit shows where the remaining peak comes from.
Working set reaches 3.09 GiB for about 4 s while the weights load, then holds
at **1.13 GiB** for the rest of the fit, including the streamed window. Private
bytes settle near 8.07 GiB, but only 1.13 GiB of that is resident. The rest
tracks the ~7 GiB of device allocations and appears to be WDDM commit backing
for VRAM. It counts against the commit limit (RAM + page file), not against
physical RAM. With streaming, the fitter's host RAM no longer depends on the
number of windows. On a 16 GB machine the remaining question is the commit
limit, which the default page file normally covers. That has not been tested
on a 16 GB machine.

The teacher cache costs 15,435,281,056 bytes (14.38 GiB, 98.3 MB per window)
of disk per AR family run. Hashing it is cheap compared with fitting. The
resumed coordinator spent about 20 s before launching native, which also
includes hashing the 7 GB base model, and about 9 s after it, with a warm
file cache.

Teacher preparation now runs once, as a separate pass of about 1.9 s per
window. The in-memory path paid the same cost at every process start: 157
windows took most of the 414 s above. A resumed recipe-4 segment prints no
`prepare_teacher` lines and goes straight to header validation.

### Estimates shown at dataset review

`ar_resource_estimate` uses these measurements:
`max(3.1 GiB load transient, 0.95 GiB + 2 × window + resident teachers)`. For
CPU fitting it adds the base and companion file sizes; that part is derived
and not measured. Teacher disk is the sum of training windows' distributions
plus small headers. For the Koan corpus this predicts 15.5 GiB (recipe 3) and
3.1 GiB plus 14.38 GiB of disk (recipe 4), matching the table. Doubling the
corpus doubles the disk and leaves host RAM unchanged.

## Parity

- **CPU, real BF16 weights** (`tests/test_native_ar_teacher_cache.py`; set
  `YUE2_AR_TEACHER_TEST_BINARY` and `YUE2_AR_TEACHER_TEST_MODEL`). LoRA and
  DoRA fixtures: three training windows plus validation and retention, KL 0.2,
  shuffled. Step-2 and step-3 states and adapters have identical tensor
  SHA-256s, sampler RNG, `metrics.jsonl` and history except wall time.
  Re-running preparation leaves files byte- and mtime-identical. A
  resume from the cached step-2 state reproduces step 3 exactly without
  teacher preparation. Wrong teacher identity, base hash or AR slot count is
  rejected before training. Both pass (372 s).
- **CUDA, real Koan corpus.** The in-memory fit, the streamed fit and the
  streamed resume from step 2 have identical step-2 and step-4 optimizer
  states, adapters and RNG. Heldout per-window scores at steps 0, 2 and 4 and
  every update's loss, CE, KL and gradient norm are equal. The resume repeats
  steps 3–4 exactly.
- **Format unit test** (`yue2-ar-teacher-cache-test`). Covers exact round trip,
  refusal to overwrite, rejection of each provenance field and of the window
  index, manifest hashes, prefix/codes contents and wrong shape, and truncation.
- **Coordinator tests** (`tests/test_ar_teacher_index.py`). Cover the SHA-256
  index, reuse without re-running native, byte/provenance/count changes,
  resumable preparation, identity immutability, unexpected files, and progress
  text.

One run used for the timeline launched native directly without
`GGML_CUDA_TF32=0`. Its losses differ in the fifth digit (TF32 matmuls, about
2× faster updates) with the same window order. It is used only for the memory
timeline, not for parity or timing.

An earlier attempt at the CUDA resume started while other GPU clients briefly
held VRAM. It placed 650 MiB in shared GPU memory and stalled in evaluation,
the known WDDM paging slowdown. It was stopped and rerun from a clean GPU.
Bits do not depend on this.

## F16 storage (assessed, not adopted)

F16 would halve the disk to about 7.2 GiB. Measured on 40 of the Koan teacher
files (30,000 rows), rounding each F32 distribution to F16:

| Quantity | Value |
| --- | ---: |
| Mean / max \|row sum − 1\| | 6.9e-5 / 3.8e-4 |
| Rows failing the trainer's existing 1e-4 normalization check | 5,764 (19%) |
| Mean / max L1 change per row | 4.9e-4 / 7.5e-4 |
| Max probability mass lost to underflow | 1.6e-4 |
| Max entropy change | 2.3e-3 nats |

The perturbation is small relative to the CE gradient, but F16 is not a drop-in
format. About a fifth of rows would fail the existing validation, so F16 would
need renormalization or a relaxed check. That is a recipe change, and it needs
a fitted heldout-CE comparison that was not run here. F32 stays the default.

## Not covered

- No full 800-update fit with the streamed cache; parity was established at 4
  updates, where any teacher difference would show at step 1.
- No DoRA run on CUDA (CPU DoRA parity passes).
- No measurement on a 16 GB host or on Linux. The coordinator records peaks
  only on Windows.
- Each family (LoRA/DoRA) prepares its own cache because each run folder is
  immutable. Sharing one cache would halve preparation and disk, but it needs
  an explicit cross-run identity.
- Teacher caches are retained with the run; there is no cleanup policy yet.
