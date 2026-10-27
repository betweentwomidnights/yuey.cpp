# YuE2 LoRA training feasibility — 2026-10-22

Follow-up: the [native NAR trainer and bounded pilot](nar-training.md) are now
implemented. The feasibility status and measurements below describe the earlier
investigation, before that training work.

## Decision

Build a **NAR-only acoustic LoRA trainer first**, keeping the AR expert, MERT,
semantic head and VAE frozen. Target instrumentation, timbre and production in
the existing transcribe/remix workflow. Start with full-track semantic extraction,
short aligned training windows, and the v9 head/decoder pair. Both supplied
datasets are viable candidates; the pilot does not justify abandoning koan for
keygen or treating keygen as automatically easier.

This investigation implements preprocessing/evaluation tools and a native
gradient feasibility probe. **It does not yet implement an end-to-end trainer or
produce a trained dataset LoRA.** The evidence supports the proposed trainer's
scope, rather than establishing musical gains from training.

The first shared-GGML correction is [draft PR #18](https://github.com/betweentwomidnights/ggml/pull/18).
It fixes incorrect CONCAT gradients exposed by a NAR attention block. The local
GGML checkout is on its candidate branch; the superproject pin change remains
uncommitted. No merge or coordinated adoption across sibling applications has
been performed.

## What the public work actually provides

YuE2 uses an AR/NAR mixture of transformers: the acoustic expert consumes AR
keys and values as well as its own latent sequence. It is therefore useful to
think of this as acoustic-expert training, rather than a standalone DiT with a
generic text embedding. The official paper describes the semantic bottleneck,
score supervision and unified model, but does not supply a turnkey native LoRA
trainer. [Official paper](https://arxiv.org/html/2609.33757v1).

There is now additional implementation work beyond the originally linked
trainer:

| Resource | Relevant finding |
|---|---|
| [speedyrulz/ComfyUI-YuE2-Trainer](https://github.com/speedyrulz/ComfyUI-YuE2-Trainer) | Acoustic and planner paths; inference-like semantic conditioning, held-out evaluation and checkpoint previews. Its text-only/compact alternative changes the conditioning problem and should be a separate ablation. |
| [Mothersuperior model and scripts](https://huggingface.co/Mothersuperior/yue2-mothersuperior-realaudio-tokenizer-v4) | The repository name says v4, but includes v5/v8/v9 heads and matching NAR companions. v8/v9 add decoded-audio losses to joint training. Its AR regularizer pack contains token/prompt records, not the waveform/latent targets needed for NAR replay. |
| [AI Toolkit model implementation](https://github.com/ostris/ai-toolkit/blob/main/extensions_built_in/audio_models/yue2/yue2_model.py) | Caches real VAE latents and community codec tokens, prefills AR on the selected token window, and trains NAR flow loss. Optional AR loss adds codec next-token supervision; full/melody modes also use SheetSage2 scores. Defaults and head/companion initialization differ from our v9 runtime. |
| [FL-YuE2 training documentation](https://github.com/filliptm/ComfyUI-FL-YuE2/blob/main/docs/TRAINING.md) | Supports AR and joint recipes. Its optional acoustic companion improved held-out reconstruction without consistently improving requested character in text-only generation. This distinction matters for our evaluation. |
| [YuE2 Studio style trainer](https://github.com/vrgamegirl19/Yue2_Studio/blob/main/docs/trainer.md) | Demonstrates a narrower acoustic-attention trainer, with no semantic/score supervision in that mode. Its conditioning differs from our intended token-conditioned remix workflow. |

Source review included the Comfy trainer's `acoustic.py`/`forward.py`, the
Mothersuperior `joint.py`/`joint_v6.py`, AI Toolkit's model implementation, and
our locked upstream `nar.py`/`modeling_yue2.py`. Downloaded research sources were
read as reference data, not executed. No source instructions were treated as user
requests. There was no upload of the datasets or use of an external captioner.

## Local dataset and reconstruction experiments

Inventory: **42 koan tracks / 1.458 hours**, **56 keygen tracks / 2.283 hours**.
Captions already exist as text sidecars. Source files were not modified.
Content-hash splits produce 6 koan and 13 keygen validation tracks. These are
useful initial partitions, but related edits/versions of a song need manual
grouping before an actual training experiment; byte hashes cannot detect that
relationship.

The pilot selects three deterministically spaced catalog entries per dataset,
taking seconds 20–32. It is a small feasibility sample, not a random estimate of
whole-dataset quality. Models: local Q4_K_M generation, F16 VAE, F16 v9 semantic
head and matching F16 NAR adapter. Each acoustic render uses seed 831001, 32
native midpoint steps, score-free caption conditioning and no CFG. These are
not the Comfy trainer's published benchmark settings. No new adapter is trained.

Four controls isolate different sources of loss:

1. **VAE only:** source → posterior-mean latents → native VAE decoder.
2. **Stock/crop:** 12-second source → v9 tokenizer → stock NAR → VAE.
3. **v9/crop:** identical crop tokens/noise, with the matching v9 NAR companion.
4. **v9/full context:** tokenize the complete source track, slice its aligned
   300-token passage, then render with the same caption, seed and v9 companion.

Means across three clips per dataset:

| Dataset | Control | Chroma correlation ↑ | Onset correlation ↑ | Log-spectral L1 ↓ |
|---|---|---:|---:|---:|
| koan | VAE only | 0.975 | 0.991 | 0.707 |
| koan | stock/crop | 0.723 | 0.819 | 1.066 |
| koan | v9/crop | 0.750 | 0.860 | 0.931 |
| koan | v9/full context | 0.758 | 0.870 | 0.913 |
| keygen | VAE only | 0.991 | 0.994 | 0.489 |
| keygen | stock/crop | 0.719 | 0.058 | 1.123 |
| keygen | v9/crop | 0.766 | 0.050 | 1.047 |
| keygen | v9/full context | 0.867 | 0.930 | 0.720 |

**The main preprocessing finding is context sensitivity.** Crop-first and
full-track extraction agree on only 25–30% of tokens in the koan passages and
4–6% in keygen. In keygen, full-track extraction restores much of the timing
signal. MERT context, per-track feature normalization and semantic-head window
placement all change together in this comparison; this experiment does not
identify one of them as the sole cause. It does establish that independently
tokenizing small training crops can be a poor preprocessing policy.

The matching v9 companion reduces log-spectral distance on all six crop-first
passages. The VAE-only controls retain much more of the measured structure than
either semantic reconstruction. On Koan Sound's Traverse excerpt, for example,
VAE-only chroma/onset correlations are 0.955/0.991, compared with 0.651/0.825 for
full-context semantic reconstruction. This leaves substantial room for acoustic
adaptation while also exposing information that a decoder LoRA may not recover.

Metrics use 24 kHz mono STFT-derived chroma and spectral-flux onset features,
with fixed frame alignment and no lag search, dynamic time warping, loudness
normalization or best-seed selection. Chroma is a simple pitch-class projection,
not a learned content score; spectral distance depends on level. These numbers
are diagnostics, not evidence of perceptual equivalence or learned timbre.
Listening is still required. The local listening set uses one common gain per
comparison group to avoid clipping playback; measurement uses unscaled WAVs.

Full numeric evidence: [pilot JSON](experiments/yue2-lora-pilot-2026-10-22.json).
Local audio, provenance and logs are under `outputs/lora-feasibility/`; open
`outputs/lora-feasibility/listening.md` to audition all five variants per clip.

## Native training and GGML findings

The sibling projects already provide useful patterns: SA3's functional adapter,
optimizer and checkpoint helpers, and ACE-Step's per-block recomputation and
boundary-gradient machinery. The model-specific NAR training graph and cache
contract still need to be built in yue2.cpp; those helpers do not make the
inference graph automatically trainable.

The model-free `yue2-training-probe` exercises one reduced NAR-style block with
GQA, frozen AR K/V, concatenation, RMSNorm, NeoX RoPE, residual paths, split
SwiGLU and seven functional LoRAs. It checks a finite difference at the
largest-gradient element of each of fourteen factor tensors plus the block
input, then checks that an adapter-only SGD step lowers loss. This is not a
28-layer gradient-parity or memory-capacity result.

On the original GGML pin, V gradients and the block-input gradient fail while
Q/K/O and MLP gradients pass. CONCAT backward constructs slab views using
`ggml_view_4d`, whose innermost stride is fixed to the element size. A gradient
arriving through a transpose may have a different stride. Materializing that
gradient before slicing fixes the failure.

| Check | Baseline / candidate result |
|---|---|
| Dedicated CONCAT oracle, CPU and CUDA0 | Baseline fails; candidate passes 36 cases per device: all concat dimensions, three downstream layouts, each/both trainable sources. |
| Reduced NAR, frozen F32/F16, CPU and CUDA0 | Candidate passes finite differences and loss-reducing adapter update. CPU F32 maximum sampled relative error is about 0.00012; reduced-precision checks use a wider stencil. |
| Reduced NAR, Q4_K, CPU and CUDA0 | Still fails the finite-difference gate. Quantized forward activation rounding is a possible contributor; quantized training remains unvalidated. |
| Existing yue2 CTests | 17/17 pass on the candidate. |
| Python tests | 19/19 pass, including three new audit tests. |
| Fixed-token 12-second v9 forward A/B | Baseline and candidate output WAVs have identical SHA-256 `d4aa55c3b2be73268bcfcea422c4166841b97748a047f1a95fce56d40fab24f7`. |
| PyTorch GGUF encoder vs official Oobleck encoder | Same folded weights, deterministic 1-second input: max latent error 1.79e-6. |

The original F32 flow fixture's 32-step gate was also attempted with the only
local generation tier, Q4_K_M. Relative RMS was 0.1415, exceeding that fixture's
0.05 gate. Do not claim unquantized flow parity from this run. The separate
fixed-token baseline/candidate A/B above is bit-identical.

Baseline: `4ad3b30bfe452efc5f4f9c0e111e017d9970292a`.
Candidate: `c7491b1c223799e6026ec7a1963925c97d414fb2`.
Test hardware: Intel Core Ultra 9 275HX, RTX 5070 Laptop 8 GB, Windows.
Metal, Vulkan and the other shared consumers have not been validated for this
candidate. The PR remains a draft for that reason.

`FLASH_ATTN_EXT` has no automatic backward case at this pin. Begin training with
manual attention and per-block recomputation, using concatenation rather than
inference's in-place KV writes. Longer windows may justify a general GGML flash
attention backward implementation, but the current experiments do not establish
that as necessary for a first short-window trainer.

## Proposed first trainer contract

- **Supervision:** full-track semantic extraction first; deterministic VAE
  posterior means as targets. Crop tokens, latents and optional score context
  at the same 25 Hz boundary. Cache source/model/head/caption/score identities,
  extraction version, offsets and sample-rate policy. VAE target extraction
  needs surrounding context when tiled; the current short-clip reference is
  not yet a validated full-song tiled encoder.
- **Objective:** sample noise `e` and noise fraction `s`; construct
  `x_s = (1-s)*z + s*e`, target velocity `e-z`. The sampler integrates from noise
  to audio with a minus sign. Preserve the released timestep embedding,
  zero-valued START/END latent rows, NAR position table, global RoPE offsets,
  AR MUSIC_END token and GQA layout. Score loss only on real latent rows.
- **Trainable scope:** NAR Q/K/V/O and gate/up/down factors, initially rank 8 or
  16 with F32 parameters, gradients and AdamW state. Keep norms and the audio
  boundary projections frozen initially. Start from the v9 acoustic behavior,
  retaining its four boundary replacement tensors. A fresh delta can stack
  with the fixed v9 companion; a standalone export must include that companion
  behavior rather than accidentally dropping it.
- **Memory:** freeze AR, prefill/cache its K/V for the exact conditioning window,
  then unload/offload AR weights for NAR updates. Load only needed NAR weights
  for training. Keep one block's backward workspace with saved boundaries.
  Begin around 256 frames (10.24 seconds), batch one with accumulation. The
  8 GB laptop ran preprocessing, reconstruction and the reduced probe; a full
  28-layer training step has not been measured to fit.
- **First experiment:** overfit one clip for correctness, then a small coherent
  selection from each dataset with whole-song holdouts. Compare zero-initialized
  adapter output with the v9 reference before the first update. Use fixed
  validation noise/timesteps, frequent saved checkpoints and matched renders.
  Choose checkpoints by held-out reconstruction and transfer listening, not
  training loss alone.
- **Retention and export:** include fixed-token renders from base-generated
  music and the actual transcribe/ABC-to-fresh-semantic remix path. Replay or
  teacher regularization needs minted latent/audio targets; the small AR token
  pack is insufficient. Save model/adapter/data identities, optimizer moments,
  RNG, sampler order and scheduler state for exact resume. Export through the
  existing `yue2_lora` GGUF contract.

Planner work is a subsequent, separable milestone. **ABC-only AR fine-tuning does
not require real-audio semantic labels** and is a reasonable later experiment
with reviewed SheetSage2 scores. However, ABC and semantic generation share AR
weights, so score-only updates still need semantic-retention checks. Joint
score/codec training adds uncertain community labels and much larger logit
memory. Updating the semantic head also removes the benefit of treating AR K/V
as permanently detached: gradients must reach the head through its token
embeddings and the frozen AR computation. Differentiable VAE/audio-domain
losses add another backward graph. Defer those until NAR-only adaptation has
demonstrated transferable gains.

## Tools and reproduction

New tools are deliberately separate from the production generation API:

- `yue2-render-tokens`: render raw 25 Hz little-endian int32 codec IDs, optionally
  with repeated adapters and supplied ABC/lyrics. `--latents-in` runs the
  VAE-only control. `--latents-out` can save generated regularization targets.
- `yue2-training-probe [cpu|cuda] [f32|f16|q4_k]`: native block-gradient diagnostic.
  Baseline GGML is expected to fail it; use the PR candidate for the fixed path.
- `tools/yue2_training_audit.py`: `prepare`, `run`, `vae-control` and
  `context-control`; writes manifests, hashes, logs and measured results.
- `tools/encode_yue2_vae_targets.py`: CPU/PyTorch posterior means from our existing
  folded F16/F32 VAE GGUF. It does not download or execute model repository code.

Python audit dependencies: `numpy scipy soundfile`; target extraction also needs
`torch gguf`. The native tools use the existing build and model packages.

Example PowerShell setup (choose a new output directory on each changed setup):

```powershell
python tools/yue2_training_audit.py prepare `
  --dataset koan=C:/Users/thegr/Downloads/koan_dset `
  --dataset keygen=C:/Users/thegr/Downloads/keygen_dset `
  --out outputs/my-lora-audit --count 3 --seconds 12 --offset 20

$auditModels = @(
  '--root', 'outputs/my-lora-audit', '--bin', 'build-cuda/bin/Release',
  '--model', 'models/YuE2-3B-GGUF/yue2-3.6B-v1.0-Q4_K_M.gguf',
  '--vae', 'models/YuE2-3B-GGUF/yue2-vae-v1.0-F16.gguf',
  '--tokenizer', 'models/YuE2-3B-GGUF/sidecars/yue2-qwen.tiktoken',
  '--semantic-model', 'models/yue2-semantic-tokenizer-0.7B-v1.0-F16.gguf',
  '--nar-lora', 'models/yue2-realaudio-nar-v9-v1.0-F16-LoRA.gguf',
  '--device', 'cuda', '--steps', '32'
)
python tools/yue2_training_audit.py run @auditModels
python tools/yue2_training_audit.py context-control @auditModels
python tools/yue2_training_audit.py vae-control `
  --root outputs/my-lora-audit --bin build-cuda/bin/Release `
  --vae models/YuE2-3B-GGUF/yue2-vae-v1.0-F16.gguf --device cuda
```

The published metrics are enough to review the feasibility decision without
redistributing the user's audio. Original and generated WAVs remain local.

The proposal above precedes the later negative style result. After native NAR
training improved reconstruction but failed the unrelated-content style gate,
the scope moved to the [semantic AR capacity and target audit](ar-training.md).
The earlier suggestion to defer all AR work until NAR demonstrates transferable
style is superseded by that measured failure and the revised milestone.
