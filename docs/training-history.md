# Training development history

This page condenses the dated reports written while the two-stage trainer was
developed (2026-10-22 to 2026-10-28). It records what was tried, what was
measured and what each result decided. For current behavior, see
[training](training.md), [AR training](ar-training.md) and
[NAR training](nar-training.md).

Unless an entry names the DGX Spark, measurements are from one Windows
machine: RTX 5070 Laptop GPU (8 GB), 64 GB of host RAM and 24 logical CPUs,
with `GGML_CUDA_TF32=0`. Listening
results are the user's own judgments on the cases presented. None of them is a
broad-seed or long-form pass. Raw commands, hashes, audio and per-run JSON
records stay in the local, git-ignored `outputs/` folder named for each
experiment.

## Feasibility investigation (2026-10-22)

- **Decision:** build a NAR-only acoustic LoRA trainer first. The AR expert,
  MERT, semantic head and VAE stay frozen.
- **Why:** YuE2's acoustic expert reads the AR expert's keys and values, so
  training it is acoustic-expert training rather than a standalone DiT. No
  public turnkey native LoRA trainer existed.
- **GGML fix:** the gradient probe exposed incorrect CONCAT gradients in a NAR
  attention block. The fix became
  [ggml PR #18](https://github.com/betweentwomidnights/ggml/pull/18).

## NAR pilot listening review (2026-10-24)

- **Setup:** a blind A/B of the four-track step-48 NAR adapter against fixed
  v9.
- **Result:** the adapter won six short clips clearly. The other two were near
  ties leaning toward the adapter, and the long clips were close.
- **Read as:** a fidelity gain, chiefly less high-frequency degradation. It is
  not evidence of style transfer.

## Expanded NAR pilot (2026-10-24)

- **Run:** 192 updates over 12 training tracks. Sampling visited 46 of 48
  windows, covering 471 s of distinct target audio.
- **Losses:** training and validation loss both fell.
- **Listening:** outputs stayed musically very similar; the main change was
  less high-frequency degradation.
- **Clarified objective:** the user then defined the real goal. That is a
  learned dataset style that carries over to new scores, unrelated source audio
  and standard generation, with no dataset references at inference. See the
  [style-transfer objective](yue2-style-transfer-objective.md).

## GGML 0.27 migration (2026-10-24)

The trainer moved from GGML `c7491b1` (0.17 plus the CONCAT fix) to
`b77ae790` (0.27,
[ggml PR #19](https://github.com/betweentwomidnights/ggml/pull/19)). The new
version reproduced the old one exactly:

- CONCAT backward passes on CPU and CUDA.
- On CUDA, these are byte-identical across revisions:
  - known-token predictions and gradients;
  - AdamW updates, including optimizer and RNG state;
  - four-update stochastic runs;
  - resumes, which also match uninterrupted runs.
- Actual ABC remixes are byte-identical.

## AI Toolkit and sv-billie reviews (2026-10-25)

- **AI Toolkit (read-only review of ostris/ai-toolkit `73ebae1`):** our pilot
  recipe differed materially from AI Toolkit's. Raising the rank or copying its
  950 updates would not be enough on its own. The higher priorities are label
  validation and wider data and context coverage.
- **sv-billie artist adapter (Hub revision `94fe9fb`):** it adapts both the AR
  and NAR experts. That supports a restrained AR adaptation paired with a
  matched acoustic adaptation.
- **Product policy:** the user accepted learned trigger words and descriptive
  style prompts as valid conditioning. Neutral captions are kept only as a
  diagnostic.

## NAR-only style gate failed (2026-10-25)

- **Test:** existing NAR adapters on fresh content with neutral captions:
  - a user source recording;
  - two independently authored scores;
  - standard generation.
- **Verdict:** the user reported A, B and C as essentially identical, and none
  matched the Koan style.
- **Loading checked:** all 196 projections were bound and every B factor was
  nonzero. Training and inference apply the same branch.
- **Decision:** stop scaling a recipe whose falling reconstruction loss does
  not become style. Train the semantic AR expert.

## First Koan AR pilot and score audit (2026-10-25)

- **Pilot:** two tracks of real Koan audio.
- **Result:** step 32 picked up the Koan palette and kept the supplied chords,
  with some degradation. By step 128 much of the melody was lost.
- **Audit:** existing windows carried very sparse note supervision; some
  training windows had no notes at all. This is a plausible contributor to
  content drift, not a proven sole cause.
- **Decision:** keep step 32 as the reference, and select later windows for
  note coverage.

## Prompted rank-32 Koan AR/NAR (2026-10-25)

- **Data:** 18 windows of 256 frames from 12 tracks (184 s), plus six held-out
  tracks. Training windows needed at least 8 notes, 3 pitches and 25% note
  coverage.
- **AR:** selected step 150. Held-out semantic CE fell 1.47%, while neutral
  retention rose slightly.
- **NAR:** selected step 400. Held-out flow MSE fell 21.1%.
- **Listening:** AR rank-32 step 150 balanced palette and melody better than
  rank-8 step 32. After broader listening, AR150 with NAR400 had the highest
  quality.

## Toolkit follow-up and AR200 (2026-10-25)

- **Kept:** the v9 semantic head (preferred over v4 in matched passages) and
  uniform noise sampling (preferred in a blind audition).
- **AR200:** a scored-only expansion with 32 windows of 20.48 s and per-track
  captions. It selected step 200 by held-out CE.
- **Against AR150:** on identical examples, AR200 had 0.98% lower held-out CE
  and slightly lower retention CE.
- **Listening:** AR200 picks up the triplets and sounds great.

## AR200 NAR calibration (2026-10-26)

- **Setup:** NAR400 was calibrated against frozen AR200 conditioning.
- **Result:** real held-out flow MSE fell 19.0%, while neutral retention rose
  2.1%.
- **Listening:** the calibrated NAR400 was rated better than native NAR. This
  set the pattern: freeze the chosen AR first, then calibrate NAR against it.

## DoRA comparisons (2026-10-26)

| Comparison | Losses | Listening |
| --- | --- | --- |
| NAR DoRA vs NAR LoRA, 400 updates, same AR150 | Tied (1.07932 vs 1.07936) | About the same at 100 and 400 updates |
| AR DoRA vs AR LoRA, original AR150 recipe | Both select step 150; CE tied | "AR DoRA is better" |
| NAR LoRA400 calibrated against AR DoRA150 | Held-out MSE fell | Calibrated pair added to the web UI |

- **Implementation:** DoRA adds output magnitudes and uses SA3-style full
  normalization gradients. Optimizer state carries a strict guard on the
  adapter family.
- **Exports:** generation GGUFs include `.dora_magnitude` and
  `.dora_scale`.
- **Bug fixed:** cache preparation for BF16 DoRA hit CUDA VMM allocation
  errors on 8 GB, so it ran on CPU.

## Full-corpus Koan training (2026-10-26 to 27)

- **Data audit:** the earlier adapters had trained on only 3 to 11 minutes of
  unique audio.
- **Full corpus:** 42 recordings (87.5 min), with 36 for training and 6
  recordings held out. That gave 157 training and 38 validation windows of 30
  s, covering 98.6% of the source audio. Scores were re-engraved from absolute
  note times on a 12-tick grid, and a native MIDI audit of 14,248 notes
  matched.
- **Recipe:** rank 32, LR 2e-5, KL 0.2 and 800 updates with recording-balanced
  sampling. Both LoRA and DoRA select step 500, after which validation CE
  rises.
- **Listening:** DoRA500 was preferred, and its NAR LoRA400 calibration beat
  native NAR.
- **Memory profile** (30 s windows, rank 32):

  | Family | Peak dedicated | Seconds per update |
  | --- | ---: | ---: |
  | LoRA | 6,900 MiB | 3.4 |
  | DoRA | 7,728 MiB | 3.85 |

  60 s windows spilled into shared memory and slowed to 42–100 s per update,
  so 30 s became the default.

## Windows GPU-residency slowdown (2026-10-27)

- **Symptom:** in the `kev-cuda-ooops` UI run, updates took 86 s while
  another process (ace-server) held GPU memory.
- **Unloading was not enough:** the same trainer process stayed slow after the
  other process freed its memory.
- **Fresh process:** at the same shape it ran at 3.6 s per update.
- **Diagnosis:** WDDM residency, with about 96 MiB moved from dedicated to
  shared memory.
- **Guidance:** after freeing GPU memory, resume the run rather than
  continuing the slow process. Cooperative stop and resume preserve the
  optimizer and sampler state.

## Native VAE encoder (2026-10-27)

- **Change:** VAE targets now use the C++ `VaeEncoder` and the
  `yue2-vae-encode` batch tool, which removes PyTorch from the production
  trainer.
- **Accuracy against the F32 reference** (relative latent L2):

  | Backend | Error |
  | --- | ---: |
  | CUDA | 6.95e-6 |
  | Vulkan | 7.52e-6 |
  | Full native reader and resampler path | 7.03e-6 |

- **Tiles:** 128 latent frames each, with 16 frames of context on both sides.

## Conditioning-cache optimization (2026-10-27)

- **Problem:** NAR conditioning caches loaded inference components they never
  used.
- **Fix:** an AR-only CUDA path for the worst-case 1,882-token example.

  | Measure | Before | After |
  | --- | ---: | ---: |
  | Time | 6.3 s | 2.6 s |
  | Dedicated GPU memory | 7,738 MiB | 4,318 MiB |

- **Cache format:** compact F16 caches halve disk use (412 → 206 MiB per
  window). F32 upcasts are lossless.
- **Same day:** the first web workflow shipped.

## Recipes 2 and 3: preflight and fingerprints (2026-10-27)

On the 166-window laptop corpus (26 GiB of conditioning), startup scanned every
cache twice and ran 498 evaluation forward passes before the first update.

- **Recipe 2:**
  - preflight reads headers only (0.09 s);
  - fitting does one full scan;
  - new runs evaluate the held-out split by default.
- **Recipe 3:**
  - fingerprints the encoded cache bytes (`bytes-v2`, FNV-1a 64-bit) instead
    of decoded floats, so the full audit fell from 183 s to 57 s;
  - SHA-256 still covers every byte in the coordinator, with no trust based
    only on timestamps;
  - older runs keep their recipe and fingerprint.

## Host-overhead reduction (2026-10-28)

These changes removed host work around each update with no change to numerics.
CPU and CUDA adapters, optimizer state and resumes are byte-identical before
and after.

| Item | Before | After |
| --- | ---: | ---: |
| NAR seconds per update (median) | 2.12 | 1.67 |
| AR seconds per update (median) | 3.66 | 3.35 |
| NAR inputs: F16 K/V uploaded without host upcast | 0.184 s | 0.052 s |
| NAR cache load for a new window: prefetch | 0.34 s | 0.01 s |
| AdamW and finite/norm scans: threaded | 0.27 s | 0.08 s |
| `bytes-v2` audit, 166 caches, warm | 34.5 s | 6.4 s |
| Coordinator SHA-256, same caches, warm | 13.3 s | 2.7 s |

## Recipe 4: streamed AR teacher cache (2026-10-28)

- **Problem:** the AR fitter held every window's teacher distribution in RAM,
  98 MB per 30 s window and 14.4 GiB for the full Koan corpus. It also
  recomputed them on every resume.
- **Change:** recipe 4 prepares the teachers once into immutable GGUF files
  indexed by SHA-256, then streams them during fitting.
- **Peak host memory:**

  | Measure | Before | After |
  | --- | ---: | ---: |
  | Working set | 15.53 GiB | 3.09 GiB |
  | Private bytes | 22.50 GiB | 8.19 GiB |

- **Resumes** skip teacher preparation.
- **Parity:** optimizer tensors, adapters, sampler state and losses are
  bit-identical to the in-memory path on CPU and CUDA.
- **Compatibility:** recipes 1 to 3 keep the in-memory path.

## Disk retention (2026-10-28)

- **Problem:** a 38-recording run with 30 s windows peaked at about 50 GB of
  disk. That was the NAR conditioning (26.1 GiB), the AR teachers (12.6 GiB)
  and every checkpoint (about 9.5 GB), all kept until manually deleted.
- **Change, with numerics unchanged:**
  - fits keep only step zero, the heldout-best and the latest resumable
    checkpoint;
  - the teacher cache is released after AR selection, before NAR
    conditioning;
  - publishing releases caches, optimizer states, other checkpoints, probe
    diagnostics and audio slices, unless `keep_caches` is set;
  - training refuses to start when the estimated peak would not fit.
- **Found during verification:** on Windows, a live `GGUFReader`
  memory-maps its file and blocks deletion, so state files are now read
  through a scoped helper.
- **Followed by** recipe 5 storage modes (below), which trade training time
  or VRAM for the remaining caches.

## Recipe 5: storage modes (2026-10-28)

- **Problem:** even with retention, a full-corpus run needs the larger of
  26.1 GiB of NAR conditioning and 12.6 GiB of AR teachers. Other trainers
  used by downstream apps need far less disk.
- **Change:** per-run `teacher_storage` (`cache`, `online`) and
  `conditioning_storage` (`cache`, `rolling`, `resident`), with a capability
  probe for downstream apps. See [storage modes](training.md#storage-modes).
  - The online teacher runs the common model in the AR fitter. With zero B
    factors the adapted block adds exactly zero, so it equals the cached
    teacher.
  - Rolling preparation records each cache's size, FNV-1a hash and header,
    then deletes it. The fit regenerates only the windows the native sampler
    will draw in the next segment.
  - Resident conditioning computes the frozen AR K/V inside the NAR fitter.
- **Parity:**
  - At smoke scale, every mode is bit-identical to cached. That covers:
    - the online teacher on CPU and CUDA, LoRA and DoRA, including resume;
    - rolling (six segments) and resident on CUDA;
    - the full server workflow through publish.
  - On 8 real recordings with 30 s windows at rank 32, rolling gave an
    identical history and final state. Resident matched every loss up to
    update 36, when it was stopped. A 10-update online AR fit equalled the
    cached fit.
  - One earlier online measurement differed by about 1e-5 from step 1; it did
    not reproduce.
  - On the DGX Spark (GB10, aarch64, CUDA 13), ctest, the unit tests and the
    CUDA online-teacher parity tests pass. Every mode was identical to cached,
    both on the synthetic set and on 8 real recordings with 30 s windows at
    rank 32.
- **Found during verification:**
  - A rolling segment that drew the same window twice regenerated it twice and
    failed with "cache output exists". Regeneration now deduplicates by path.
  - Resident conditioning initially differed by 1e-5. Preparation computes
    without flash attention, and the resident model now does the same.
  - Resident conditioning on 8 GB with 30 s windows spills: about 188 s per
    NAR update against 1.48 s cached.
    - The fitter holds the NAR expert (2.7 GiB) and the AR expert (3.3 GiB),
      which don't overlap, plus both workspaces and the K/V.
    - Four-second windows fit (0.47 s against 0.28 s).
  - With a relative `--out`, rolling regeneration failed on the Spark: the
    native tool runs from its own folder and could not find the plan.
    Regeneration now passes only absolute paths.
- **Costs on the Spark** (30 s windows, rank 32, per update against cached):
  - online teacher: 3.86 s against 2.85 s;
  - rolling: 1.21 s, the same as cached, plus regeneration (fit 215 s against
    86 s);
  - resident: 1.91 s against 1.21 s, but it skips the 50 s conditioning
    preparation, so total NAR time roughly matches cached.
- **Verification tool:** `tools/verify_training_storage.py` repeats a cached
  server run in every mode and compares the results.
- **Not yet done:** a rolling disk measurement on a dataset large enough to
  show the saving.

## Native workflow, stage 1 (2026-10-28)

- **Problem:** training needs Python with numpy, soundfile and gguf to
  orchestrate the native tools. gary4local installs native packages only, so
  it could not train. See [native workflow](training.md#native-workflow).
- **Change:** `yue2-train-run`, which will replace the Python stage by stage.
  Stage 1 ports run settings, dataset inventory, the review estimates,
  capabilities, provenance hashing and the caption, window and score rules of
  `ar_dataset.py`. SHA-256 is native; the WAV parser now has a header-only
  reader the inventory uses. The server still launches the Python.
- **Parity:** equal to the Python, values and JSON types, on:
  - fuzzed cases: 6,000 captions, 6,000 run requests, 2,000 window plans,
    3,000 transcriptions, 1,500 datasets' estimates, every chord and pitch;
  - a mixed WAV/FLAC dataset (BOM, CRLF and accented names, four sample
    formats) and every inventory rejection;
  - every run, dataset and transcription under `outputs/`: 7 runs, 4 datasets
    and 570 transcriptions.
- **Found:** 38 of the 536 `normalized.abc` files in `outputs/` differ from
  today's engraving, and from the Python's equally. They were written on
  2026-10-26, before chord inversions were engraved; that is the only
  difference.

## Native workflow, stage 2: preparation (2026-10-28)

- **Change:** `yue2-train-run prepare-semantic`, `prepare-scores`,
  `prepare-targets`, `prepare-conditioning` and `materialize` replace the three
  preparation scripts on the product path. See
  [native workflow](training.md#native-workflow).
  - Identities hash the manifests they depend on. For a run to move between
    the implementations, the native side writes those manifests byte for byte
    as the Python did. Its JSON objects now keep insertion order, and its
    writer matches `json.dumps` with Windows line endings. `latent_rms` uses
    numpy's pairwise summation, checked against numpy on 45 arrays.
  - The Python recorded digests of its own scripts, which follow the
    checkout's line endings. The native side records the committed LF digests.
- **Parity:** `tools/check_native_preparation.py` prepared three real 20 s
  clips (one a 24-bit FLAC) with 8 s windows twice in the same run folder:
  review, cache and rolling conditioning, and a rolling regeneration. All 212
  compared files were equal: manifests, 9 examples' scores, prefixes and
  MIDI, VAE targets, and 11 conditioning caches. `source.wav` samples were equal; its header and
  the logs and timings are not compared.
- **Found:**
  - `yue2-semantic-tokenize` read WAV only, though inventory accepts FLAC, so
    a FLAC dataset failed at its first semantic cache in both
    implementations. It now reads either, through the decoder uploads use.
    A 20 s real clip gave identical codes as a 24-bit WAV (before and after
    the change) and as a 24-bit FLAC.
  - A rolling regeneration run without `GGML_CUDA_TF32=0` gives different
    bytes and fails its check. `train_nar_run.py` sets it before calling
    `materialize()`; the native `materialize` now sets it itself.

## Native workflow, stage 3: fitting (2026-10-28)

- **Change:** `yue2-train-run fit-ar` and `fit-nar` replace `train_ar_run.py`
  and `train_nar_run.py`. They build the trainers' command lines and run
  identities, prepare and verify the AR teacher cache, run rolling NAR fits
  in regenerated segments, prune checkpoints, and select the heldout-best
  checkpoint. See [native workflow](training.md#native-workflow).
  - The JSON writer now writes every number as Python re-serializes it, so a
    trainer's lexeme copied into `selection.json` matches.
- **Parity:** `tools/check_native_fitting.py` prepared and trained three real
  20 s clips (one FLAC) through `Workflow.prepare()` and `Workflow.train()`
  with each implementation in the same run folder: 6 AR and 8 NAR updates,
  rank 4, evaluation every 2 updates.
  - LoRA, cached teachers and conditioning: 225 compared files equal.
  - DoRA, online teacher, rolling conditioning in 4 segments: 220 equal.
  - Equal files include every checkpoint and state, pruning's survivors
    (step 0, best and latest), the teacher index, selections and prompts.
    Logs, progress and process records, timings and peak readings are not
    compared.

## Native workflow, stage 4: publication (2026-10-28)

- **Change:** `yue2-train-run publish` and `release-caches` replace
  `publish_training_adapter.py` and `Workflow.release_caches()`. The style
  GGUF is written directly in gguf-py's layout: metadata in insertion order,
  then tensor infos, then each tensor padded to 32 bytes.
- **Parity:** `tools/check_native_publication.py` on both stage 3 runs: the
  published GGUF (34.9 MB LoRA, 39.5 MB mixed LoRA/DoRA) and its sidecar were
  byte-equal, republishing returned the same record, and releasing caches
  left the same files and released the same bytes (0.92 and 0.46 GB).

## Native workflow, stage 5: switch-over (2026-10-28)

- **Change:** `yue2-train-run workflow --run RUN --action ACTION` replaces
  `training_workflow.py`'s `Workflow` and `main()`, and `yue2-server` now
  launches it instead of Python. `yue2-train-run setup` replaces
  `setup_training.py`; the config it writes has no `python` or `workspace`.
  The standalone package no longer carries the Python scripts.
  - The driver proves ownership with the identity the server records (a new
    owner-test assertion checks the two agree) and, on Windows, keeps its
    children in a job that ends with it.
  - Resuming a run the Python started keeps the script digests that run
    recorded, CRLF or LF, so its files stay what the Python would write.
- **Checks** (`tools/check_native_workflow.py`, three real 20 s clips, one FLAC):
  - **Parity:** the Python and native workflows took the same run through
    prepare, train and publish. All 200 compared files and the final state
    were equal, and the published style was byte-equal. The released-cache
    total differs by 396 bytes, the header difference of 9 `source.wav`
    files; libsndfile's header records a write time, so it is not compared.
  - **Resume:** the Python workflow, with its own CRLF digests, was stopped
    during scores and, in a second run, during NAR conditioning. The native
    workflow resumed both, keeping the recorded digests, and published both.
  - **Server:** a run created, trained and published through `yue2-server`'s
    training API with a native config; the style appeared in `/loras`.

## Native workflow, stage 6: Python removed (2026-10-28)

- **Change:** the Python training scripts are gone: `training_workflow.py`,
  `setup_training.py`, `requirements-training.txt`, `ar_dataset.py`,
  `training_provenance.py`, the three preparation scripts, the two fit
  coordinators and `publish_training_adapter.py`. So are the parity harnesses
  that compared against them (stages 1 to 5 above record their results) and
  the CLI's `parity` command.
  - The Python tests of their behaviour are now `yue2-training-native-test`:
    progress reading, evaluation history, pruning (including a file Windows
    holds open), selection, the teacher cache (with the test binary standing
    in for the AR trainer), target matching, cache checks, preflight, rolling
    regeneration (standing in for the NAR trainer), cache release, the disk
    check, run ownership, progress-file watching, settings and inventory. With
    the built tools it also checks the score MIDI roundtrip, the NAR trainer's
    progress events and, given a VAE model, target resume.
  - `verify_training_storage.py` drives `yue2-train-run`;
    `yue2_training_audit.py` and `benchmark_nar_cache.py` carry the few helpers
    they used.
