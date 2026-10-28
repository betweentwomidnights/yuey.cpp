# Native trainer host overhead

This change follows the [recipe 3 optimization](training-final-optimization-2026-10-27.md).
It removes host-side work around each NAR and AR update without changing any
numerics. Recipe and fingerprint versions are unchanged. Existing requests,
checkpoints and resumes keep their identities. Every integrity check remains:
SHA-256 still covers every byte, and there is no timestamp-only trust path.

History records now carry `inputs_seconds` (input validation and upload) and
`optimizer_seconds` (gradient download, finite/norm scans, AdamW and parameter
upload). The baseline below is b4f5a14 with only those fields added.

## Measurements

The Windows RTX 5070 Laptop (8 GB) system was used with `--threads 6` and TF32 off.
Both binaries ran under one machine GPU lock with no other GPU process. The NAR
fixture is the recipe 3 GPU fixture: four real 750-frame F16 caches, rank 32,
1900 AR slots, heldout evaluation every 2 updates, 6 updates. The AR fixture uses
four real windows, maximum prefix 1149, rank 32 LoRA, KL 0.2 and 6 updates.
Medians are per update unless noted.

| Item | Measure | Before | After |
| --- | --- | ---: | ---: |
| 1. NAR F16 K/V | `inputs_seconds` | 0.184 s | 0.052 s |
| | Cache load without prefetch (first update) | 0.326 s | 0.181 s |
| | Peak GPU memory | 4382 MiB | 4170 MiB |
| 2+3. NAR evaluation (2 windows × 3 σ) | Wall time per evaluation | 3.63 s | 3.10 s |
| 3. NAR prefetch | `cache_load_seconds`, new window | 0.34 s | 0.01 s |
| 4. Threaded AdamW/scans, NAR | `optimizer_seconds` | 0.267 s | 0.083 s |
| 4. Threaded AdamW/scans, AR | `optimizer_seconds` | 0.269 s | 0.086 s |
| 5. AR teacher validation once | `inputs_seconds` | 0.073 s | 0.045 s |
| 6. Native bytes-v2 audit, 166 caches, 28.1 GB | Page-cache warm | 34.5 s | 6.4 s |
| 6. Coordinator SHA-256, same 166 caches | Page-cache warm | 13.3 s | 2.7 s |

Combined NAR update time (cache load plus update) fell from 2.12 s to 1.67 s
(−21%). The AR update fell from 3.66 s to 3.35 s (−8%); its remaining time is
GPU compute. The four-cache native validation fell from 1.69 s to 0.45 s. The
first, cold-cache bytes-v2 audit took 57.0 s, matching the recipe 3 record.

Items 2 and 3 are measured together in evaluation: about 0.26 s saved per
window, from two fewer K/V uploads and an overlapped cache load. The
first-update cache load isolates item 1's removed F16 decode. Prefetched loads
isolate item 3.
These short fixtures show where host time went. They do not forecast
full-corpus throughput.

## What changed

1. **NAR F16 conditioning.** Compact caches stay F16 in host memory and in the
   persistent K/V inputs; the graph casts them to F32, which is exact. F32 or
   mixed datasets keep F32 inputs, so F16 samples are widened on the host as
   before. Each head is uploaded directly to its AR slot range. Only slots that
   an earlier, longer sample filled are zeroed again. Positions and mask upload
   only when the AR length changes. Finite checks run on the half data in
   parallel. The decoded-v1 identity decodes half data with the same conversion
   to mix identical F32 bits.
2. **Evaluation uploads.** `NarTrainer::evaluate(sample, noise, sigmas)` uploads
   a window's conditioning once for its three noise levels.
3. **Prefetch.** Streaming stores load the next training window during an update,
   and the next evaluation window during evaluation. The next index comes from
   *copies* of the sampler and RNG, so saved states are unchanged. The store
   repeats its size/mtime checks before and after the background load and again
   when the sample is consumed. At most two decoded samples are resident.
4. **Optimizer.** Gradients download into existing buffers. Finite checks and the
   global norm overlap the next layer's backward pass, each scan waiting for the
   previous one, so the double sum keeps its serial order. AdamW is elementwise
   over fixed spans on `--threads` workers, with the former expression and types.
5. **AR teacher.** `prepare_teacher` validates each distribution completely once,
   with the same finite, nonnegative and row-sum checks and messages, and marks
   the sample. Unmarked distributions are still validated on every use;
   `validate_teacher()` marks one supplied from elsewhere.
6. **Hashing.** Bytes-v2 fingerprints are computed per file in parallel and mixed
   in the original order. Each file's size/mtime is recorded before hashing and
   rechecked before use, so `nar-cache-bytes-fnv1a64-v2` identities are unchanged
   (`a6e367b97c64024f` for this corpus, as before). The coordinators hash files
   concurrently with `digests()`; every byte is still hashed.

## Parity

Baseline and candidate binaries were compared exactly. The comparison covered
every GGUF tensor in adapters and states, including all 1,176 optimizer tensors
per rank-32 state, plus RNG, identity and step metadata. It also covered every
non-timing history and metric field: losses, gradient norms, sample indices,
sigmas and heldout/retention scores. Resumed continuation from step 2 also matches.

- CUDA: NAR bytes-v2 with resume, NAR decoded-v1, AR LoRA with KL and resume.
  The candidate's NAR step-2 state and its step 0/2 heldout/retention scores
  also equal the 2026-10-27 recipe 3 reference run byte for byte.
- CPU, real weights with synthetic random caches: F16 and mixed F16/F32
  datasets, streaming and in-memory, full and heldout scope, decoded-v1 and
  bytes-v2, NAR DoRA with sigmoid noise, unshuffled sampling, probe diagnostics,
  and AR LoRA/DoRA with KL and resume.

New model-free tests cover prefetch consumption and stamp rejection, parallel
fingerprints against serial ones, and parallel SHA-256 order.
Portable evidence is in the [experiment JSON](experiments/yue2-training-host-overhead-2026-10-28.json).
Local logs, the harness and both runtimes are in `outputs/training-host-overhead-2026-10-28`.
