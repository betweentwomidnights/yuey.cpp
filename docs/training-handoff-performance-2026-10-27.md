# Training handoff: duplicate validation and evaluation scope

The Windows 8 GB laptop run contains 166 real windows: 138 training and 28
heldout, with 26.14 GiB of frozen NAR conditioning. The old NAR preflight took
189.83 seconds natively, followed by another complete cache scan in fitting.
Initial full evaluation then processed 498 forward passes before update 1.
Neither a full scan nor an initial evaluation is an optimizer update.

## Measured cache work

CPU-only audits of the actual corpus, while the user's GPU run continued:

| Work | Wall time |
| --- | ---: |
| Inspect all 166 GGUF headers, recover maximum context (1900 slots) | 0.094 s |
| Decode and validate one real cache without the corpus fingerprint | 0.422 s |
| Decode, validate and fingerprint all 166 caches | 183.140 s |

The complete audit reproduced the cache fingerprint in the existing step-1
optimizer state exactly (`81eaf929f8e20ed3`). The audit does not load neural
weights or run GPU kernels. SHA-256 checks in the Python coordinator are
additional to these timings and remain enabled for every cache.

Recipe 2 preflight reads every header after checksum validation, then uses the
first training sample at the maximum context size across **all** splits. It
therefore tests the same padded graph shape and gradients as the old preflight,
without repeating the corpus-wide decode/fingerprint pass. The full fitter
retains all original content, provenance, shape and mutation checks. The
remaining roughly three-minute decoded fingerprint on this corpus still occurs
at fitting/resume; no timestamp-only trust path or checksum receipt is added.

The new `--validate-only` native mode reports this audit independently of model
loading, gradients and evaluation. It computes the same cache identity as fitting.

## Evaluation scope

Checkpoint selection uses recording-balanced heldout loss. Full training-set
evaluation is a diagnostic; it is additional to the loss from optimizer updates.
New UI jobs evaluate **every heldout and retention example** by default. The
advanced Validation scope control can retain full training-set diagnostics.

For this corpus each NAR evaluation becomes 28 × 3 = **84 forward passes**,
instead of 166 × 3 = 498, removing 83.1% of evaluation passes and their cache
loads. This percentage describes evaluation work, not total training speed.
AR omits the analogous full training-set diagnostic. Teacher preparation,
training samples, noise schedules, optimizer updates, checkpoint cadence and
heldout/retention scoring remain unchanged.

## Validation and compatibility

Real-weight CPU fixtures cover AR LoRA, AR DoRA and NAR LoRA. Heldout-only and
full evaluation produce identical heldout/retention scores, optimizer tensors
and sampler/noise RNG state. NAR resumed continuation also matches. Full-corpus
and single-sample preflight diagnostics have identical tensor bytes when padded
to the same maximum context. These are small correctness fixtures, not GPU
throughput benchmarks. The NAR fixture took 23.657 s with full evaluation and
15.750 s with heldout evaluation, including loading and three updates.

Header tests include a holdout longer than every training example and reject
incompatible frame counts. New server-created requests are pinned to recipe
version 2; both training configs record their evaluation scope, and NAR records
its preflight policy. Missing version means the legacy recipe, preserving
existing request/config identities and executable pinning. Native research CLIs
still default to full evaluation unless `--eval-scope heldout` is supplied.

Raw local artifacts are in `outputs/training-handoff-2026-10-27`. Portable timings
and assertions are in the accompanying experiment JSON. No GGML source change
is needed. Remaining candidates are a versioned native byte-hash path to avoid
decoded-float fingerprinting, persisted AR teacher distributions on resume, and
overlapping cache reads with training. They require separate integrity/resource
measurements; this change does not enable them.
