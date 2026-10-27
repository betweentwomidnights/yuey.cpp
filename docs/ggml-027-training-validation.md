# GGML 0.27 native NAR training migration

Date: 2026-10-24. This validation uses the local native NAR trainer, including
the uncommitted October 22 pilot implementation. The public pointer-only
[Yuey PR #34](https://github.com/betweentwomidnights/yuey.cpp/pull/34) does not
itself contain that trainer.

## Revisions and scope

- Training baseline: GGML `c7491b1c223799e6026ec7a1963925c97d414fb2` (0.17 plus
  the noncontiguous CONCAT-gradient correction).
- Candidate: GGML `b77ae790e628270af46a8af1eca38441c23f1252` (0.27), the exact
  merged [shared-fork PR #19](https://github.com/betweentwomidnights/ggml/pull/19)
  revision selected by Yuey PR #34.
- Application source starts at `179138784b4526d24eab868d3083f1dd07141c5e`, with
  the local trainer and feasibility changes. The trainer sources are identical
  for both builds; their hashes are recorded with the binary identities.
- Windows, RTX 5070 Laptop 8 GB, CUDA 12.8, MSVC 19.44. Both migration runs set
  `GGML_CUDA_TF32=0`. This precision setting is explicit and differs from the
  original pilot's inherited default setting.

The candidate already includes the CONCAT correction (cherry-pick
`56efc47a`). Do not apply the original fix a second time. The backward graph,
gradient lookup and graph-reset interfaces used by this trainer remain available.

## Evidence

Local evidence is under `outputs/ggml027-validation/`. It retains commands,
return codes, elapsed times, executable and DLL SHA-256 identities, raw gradient
diagnostics, independent PyTorch reports, native checkpoints, and float WAVs.
The old `build-cuda/bin/Release` binaries are retained as the baseline. The new
build is `build-ggml027-cuda/bin/Release`; use that directory for subsequent
experiments at the refreshed pin.

The compact portable results are in
[`experiments/yue2-ggml-027-validation-2026-10-24.json`](experiments/yue2-ggml-027-validation-2026-10-24.json).

| Check | Result |
|---|---|
| CPU / CUDA CONCAT backward | 36 independent cases pass on each backend at both revisions |
| CUDA known-token initial predictions / all 392 factor gradients | Complete diagnostic GGUF byte-identical across revisions |
| CUDA known-token gradients after an update | Complete diagnostic GGUF byte-identical |
| CUDA 256-frame, rank-8 real-audio/v9 predictions / gradients | Complete diagnostic GGUF byte-identical |
| CUDA default TF32 setting, additional known-token probe | Complete diagnostic GGUF byte-identical |
| First AdamW update | Adapter and full optimizer/RNG state byte-identical across revisions |
| Four-track stochastic training, four updates | Adapter and full optimizer/RNG state byte-identical across revisions |
| Resume after update 2, continue to update 4 | Adapter and full state byte-identical to the uninterrupted run, at each revision |
| Resume original pilot update 24, continue to update 26 | Adapter and full state byte-identical across revisions |
| CPU known-token probe | Predictions identical; aggregate gradient relative L2 difference 5.88e-7, maximum absolute difference 4.89e-9 |
| Actual ABC remix on both held-out passages, v9 and step 48 | All four WAVs and semantic streams byte-identical across revisions |
| Application / Python regression suites | 17/17 CTests and 19/19 Python tests pass |

CUDA tensors, not just printed losses, were compared. State comparisons include
all 392 factors, 784 optimizer moment tensors and the serialized RNG state.
The independent PyTorch oracle also passes for initial, updated, real-audio/v9
and CPU probes. Candidate aggregate gradient errors against PyTorch are 0.799%,
0.690%, 0.772% and 0.0522%, respectively. The first AdamW update differs from
PyTorch by at most 7.45e-9 when supplied identical native gradients.

The CPU comparison is not bit-identical. The candidate is a fresh native build
with AVX2 enabled; the old build's cached CPU options had AVX2 disabled. The
small gradient difference is reported rather than waived as exact equality.
Both independent CPU oracle checks pass the existing, unchanged tolerances.

No trainer math or additional GGML patch was needed. The experiment helpers
now record shared-library hashes and precision environment alongside executable
hashes; this prevents a replaced GGML DLL from silently reusing a previously
rendered preview or conditioning-cache identity. Old output directories retain
their original provenance; choose a fresh label/directory for new experiments.

## Reproducing the numerical checks

The full local driver and exact per-command argument arrays are retained in
`outputs/validate_ggml027.py` and the evidence directory. Its inputs reuse the
original frozen caches intentionally, isolating backend execution from changes
in preprocessing. The two builds have separate binaries and DLLs.

An individual initial probe and independent oracle can be run with:

```powershell
$env:GGML_CUDA_TF32 = '0'
build-ggml027-cuda/bin/Release/yue2-train.exe `
  --model models/YuE2-3B-GGUF/yue2-3.6B-v1.0-BF16.gguf `
  --sample outputs/nar-training/known-tokens.gguf `
  --rank 4 --device cuda --identity ggml027-known-control `
  --probe-only --out outputs/ggml027-probe-new
.venv/Scripts/python.exe tools/check_nar_training_reference.py `
  --model models/YuE2-3B-GGUF/yue2-3.6B-v1.0-BF16.gguf `
  --diagnostics outputs/ggml027-probe-new/probe.gguf `
  --out outputs/ggml027-probe-new/reference.json
```

The output directory must be new. For the real-audio probe, use the 256-frame
pilot caches, rank 8 and the same fixed v9 adapter used when preparing them.

## Scope limits and next auditions

These are bounded migration controls, not evidence of long-run training
convergence, perceptual improvement or cross-platform bit identity. The tests
do not expand the trainer's supported backends to Metal or Vulkan, nor remove
the previously documented CPU cached-inference limitation. The unchanged
step-48 adapter remains stacked after v9.

After these gates passed, the next listening experiment uses the refreshed
build: eight excerpts across three seeds, including four new Koan holdouts and
two keygen tracks, plus 30/60/120-second comparisons on a new Koan holdout.
Two excerpts are the previous validation tracks and must not be presented as
new holdouts. Scores are fixed, semantic streams must match within each pair,
and blind listening copies use matched RMS plus a shared peak-prevention gain.
The actual audio and final completion status are under
`outputs/nar-remix-eval-2026-10-24/`; the original pilot audio is untouched.

All 54 renders completed on the RTX 5070 Laptop 8 GB, including both
120-second variants. Every one of the 27 A/B pairs has identical semantic
streams. The exported listening copies pass finite-signal, duration,
pairwise-RMS and peak-headroom checks. This establishes execution and a
controlled listening set, not a subjective preference. The portable rendition
inventory is
[`experiments/yue2-remix-evaluation-2026-10-24.json`](experiments/yue2-remix-evaluation-2026-10-24.json).

The later [seed-831001 listening review](nar-listening-review-2026-10-24.md)
records the user's feedback and decoded identities separately from these
technical migration checks. A later four-clip seed-123 spot-check is also
recorded there; the other pairs remain unrated.
