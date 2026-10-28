# Final training optimization before review

Recipe 3 replaces the native NAR startup audit's decoded-float fingerprint with
a versioned fingerprint of every encoded cache file. Cache headers are checked
for model/adapter provenance, tensor types, dimensions and payload bounds. Every
payload byte is then read. Streaming samples are decoded only when consumed.
The coordinator retains its SHA-256 verification of every cache before and after
fitting. There is no timestamp-only cache receipt or skipped integrity pass.

Both native fingerprints are 64-bit FNV-based identities; SHA-256 remains the
cryptographic provenance check. The new identity covers encoded metadata and
padding as well as tensor bytes. F32 and F16 encodings with equal decoded values
therefore have distinct new identities. Its explicit checkpoint marker is
`nar-cache-bytes-fnv1a64-v2`; the old `nar-cache-fnv1a64-v1` mode remains the
native CLI default for research/legacy use. The new flag is
`--cache-fingerprint bytes-v2` with a prepared dataset.

New server-created requests pin recipe 3. Older requests retain their original
recipe, training configs, executable and fingerprint. No existing run is migrated.
The preflight policy and heldout-only default from recipe 2 are retained. Mismatched
fingerprint versions are rejected by native resume before any optimizer update.

## Measurements

Same 166-window, 26.14 GiB corpus on the Windows RTX 5070 Laptop system:

| Native startup audit | Wall time |
| --- | ---: |
| Previous complete decoded audit | 183.140 s |
| Complete encoded-file audit | 57.078 s |

This is about 3.2× faster for this audit component. Coordinator SHA-256 passes,
model loading, probes and evaluation are additional work. No neural math or GGML
kernel changes are involved.

Real-window CUDA checks use four existing caches, 750 frames, maximum AR context
1900, rank 32 and three updates, with checks at updates 2 and 3. They reserve the
device through the existing worker registry and leave the user's published model
and caches unchanged. All 1,176 optimizer tensors and complete sampler/noise RNG
states match exactly across full/heldout evaluation and decoded/encoded auditing.
Heldout and retention metrics match, and encoded-mode resumed continuation matches.
An edited payload with preserved file size/mtime and a wrong-mode resume both
fail before history/updates begin.

| CUDA subset run, including startup and evaluation | Wall time | Peak total GPU MiB |
| --- | ---: | ---: |
| Full evaluation + decoded audit | 48.141 s | 4632 |
| Heldout evaluation + decoded audit | 31.593 s | 4632 |
| Heldout evaluation + encoded audit | 27.328 s | 4632 |

This short fixture demonstrates savings, not a forecast of full-corpus run time.
Median cache loading was about 0.4 s per update, versus about 2.1 s for the native
update. History now records cache-load, update and combined durations separately.

## Checks and remaining scope

Model-free C++ tests exercise header/payload bounds, provenance, file edits with
restored timestamps and mutation guards; these run in the ordinary CTest matrix.
Optional native Python integration covers F16/F32 representation identities,
truncation, incompatible shapes, unsupported policies and real-weight continuation.
The optional scope test supports `YUE2_SCOPE_TEST_DEVICE=cuda` as well as CPU.

Portable evidence is in the accompanying experiment JSON. Raw local logs and
states are in `outputs/training-final-2026-10-27`. The GPU reservation is released
after checks. Teacher-distribution persistence and asynchronous cache prefetch
remain separate follow-ups: they add disk-space and concurrency contracts and
are not needed to land this bounded optimization.
The subsequent [host-overhead change](training-host-overhead-2026-10-28.md)
adds window prefetch and other host-side reductions with exact parity.
