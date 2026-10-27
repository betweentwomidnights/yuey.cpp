# Training conditioning optimization and first web workflow

Measured on the same 8 GB RTX 5070 Laptop GPU / 64 GB host machine, with GGML
`b77ae790e628270af46a8af1eca38441c23f1252` and TF32 disabled. No GGML source
change was needed. The expensive cache stage was loading inference components
it never used; removing those allocations was a larger first win than changing
kernels. The CPU VAE posterior encoder remains a separate preprocessing stage.

## Measured conditioning preparation

The real worst-prefix Koan example has 1,131 prefix tokens and 750 audio frames,
for 1,882 AR conditioning tokens. All rows below include process startup and
writing a complete sample cache. WDDM process memory is sampled, so reported
peaks are observed values rather than a guaranteed allocation maximum.

| Path | Seconds | Dedicated MiB | Shared MiB | Cache MiB |
| --- | ---: | ---: | ---: | ---: |
| Existing full-model CPU, F32 cache | 36.781 | — | — | 411.88 |
| Existing full-model CUDA, F32 cache | 6.281 | 7,738 | 278 | 411.88 |
| AR-only CUDA, F32 cache | 2.641 | 4,318 | 74 | 411.88 |
| AR-only CUDA, compact F16 cache | 3.219 | 4,426 | 74 | 206.04 |

The compact CUDA path is about **11.4× faster than the previous CPU cache
preparation** on this example, with substantial dedicated-memory headroom.
The 74 MiB shared usage is the normal process baseline seen on this machine.
This is cache preparation, not an 11× improvement in gradient-update speed.

The native batch worker also reuses one loaded AR model across the dataset.
Two worst-prefix compact caches took 6.437 seconds total; native per-item times
were 3.776 seconds cold / 2.482 seconds resident. On the smaller 319-token UI
fixture, cold/resident times were about 1.7 / 0.23 seconds. These individual
measurements do not substitute for a full 197-example timing run.

Only AR embeddings/layers/norm are loaded, in their original tensor types;
unused NAR experts, audio boundaries and the output vocabulary head are omitted.
The selected weight buffer is 3,409.74 MiB versus about 6,924.98 MiB for the full
generation model. The conditioning graph skips output logits and their final
dependencies. Generation APIs reject conditioning-only models explicitly.

## Numerical and integrity checks

- AR-only CPU conditioning matches the legacy full CPU payload exactly.
- AR-only CUDA conditioning matches the legacy full CUDA payload exactly when
  using the default unchunked/manual attention graph.
- The inference K/V cache already stores F16. Exporting its original values
  directly saves half the K/V disk traffic, without quantizing new information.
  The writer rejects any value that cannot roundtrip exactly. Restored compact
  caches match the F32 payload exactly.
- Resuming the established CPU NAR fixture from step 2 through step 4 with
  compact caches produced **all 1,176 optimizer-state tensors byte-identical**
  to the original F32-cache run, including sampler/optimizer continuity.
- CPU and CUDA are **not** bit-identical to each other: relative L2 differences
  were 0.00328 for K and 0.00510 for V. Cache identities pin backend, executable,
  shared libraries, precision, prefill and storage profile. Old CPU caches and
  experiments remain separate and unchanged.
- Bounded 256-token prefill and fused attention were explored. They change
  floating-point rounding (roughly 0.3–0.5% K/V relative L2 here), and provided
  no compelling advantage over the now-small default graph. They remain
  explicit CLI experiments, not the product default.
- Each completed native cache is checked against actual prefix/code/clean
  payload bytes, shape/type, finite conditioning, base and frozen adapters,
  then committed with a SHA-256 sidecar. A cache missing its committed checksum
  after a crash is preserved as an unregistered artifact and recomputed. The
  recovery test reproduced identical bytes and rejected injected nonfinite K/V.

The original full-run cache set was about 58.15 GiB. Compact representation
would bring the same K/V payloads to roughly half that size; the historical
files are retained. A new numerical/cache profile uses a new output directory.

`tools/benchmark_nar_cache.py` reproduces the real-weight comparison from a
completed AR job, VAE target index and selected AR adapter. `--full-cuda` opts
into allocating the old full model. Run it with inference idle. Portable
[measurement evidence](experiments/yue2-training-optimization-2026-10-27.json)
contains timing, memory, parity and smoke-run results.

An initial benchmark build accidentally linked an older static library because
MSBuild's overridden `OutDir` did not update CMake's explicit library link path.
Those provisional results were discarded. The final measurements verify the
selected loader is present and reports its reduced weight allocation. Use a
separate configured build tree for repeatable release builds.

## Web workflow validation and limits

See [training setup and recipe](training.md). Real native tools completed two
small end-to-end fixtures: AR LoRA4/NAR4 and AR DoRA12/NAR12, both rank 4, with
recording-level holdouts, actual sidecar captions, semantic/symbolic/VAE
preparation, heldout checkpoint selection, compact GPU caches and paired export.
The browser exercised preparation, training and explicit publication; the API
rejected generation while the training worker held the device. Stop at the
AR/NAR handoff and resume completed successfully.

The exported DoRA/NAR pair rendered a finite, nonzero, 48 kHz stereo smoke WAV
through the existing inference API with the required companion supplied.
Catalog resolution now supplies that companion automatically, including when
the style folder differs from the installed model folder, and rejects an
incompatible fixed prefix. Native tests cover companion relocation, fresh-owner
adoption, PID creation identity, path guards and retaining ownership after stop.

After automatic approval review blocked server process control, the user
manually restarted the updated HTTP server. Durable run history was recovered;
automatic versus explicit companion rendering produced byte-identical WAVs.
A further full LoRA4/NAR4 workflow passed the committed-ownership handshake,
current cache guards, browser publication, display name and caption-pool checks.
This caught and fixed the Windows venv redirector/worker PID distinction before
finalizing the draft. Restart during a live GPU worker remains covered by
native fresh-manager adoption tests rather than an actual interrupted-server
exercise.
The small fixtures establish orchestration, not musical quality or rank-32/30s
resource limits. Those limits come from the prior full training experiment.

The first UI recipe does not yet synthesize neutral replay/retention controls.
Its defaults and combined UX require review and broader listening before a
release. Neither SA3 nor gary4local is modified; main is not merged and no
release is published by this work.
