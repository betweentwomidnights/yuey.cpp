# Validation

Development is intentionally parity-driven. A feature is not advertised as
working merely because a graph compiles.

## Current checks

- MSVC Release CPU build against the pinned shared GGML revision.
- Native WAV decode, multichannel downmix, and 48 kHz to 24 kHz resampling.
- MERT2 log-mel probes against the official Torch/Torchaudio implementation;
  tolerance is 0.002 dB.
- SheetSage2 v1 token range and constrained grammar state transitions.
- Synthetic safetensors-to-GGUF round trip, including every one of the 96
  MERT2 attention LoRA merges.
- Synthetic YuE2 generation-package conversion on both Windows and the ARM64
  YuE2 container, including byte-exact BF16 preservation, exact Oobleck
  weight-normalization folding, sidecar copying, and GGUF re-reading.
- Synthetic PEFT LoRA conversion validates canonical target names, complete
  A/B pairs, uniform rank/alpha metadata, F16 storage, and unsupported-target
  rejection. A native functional test checks exact dense LoRA math, zero-scale
  identity, immutable base weights, and base-fingerprint rejection. The same
  functional graph passes explicitly on GB10 CUDA. A converter-produced rank-1
  F16 adapter, fingerprint-bound to the released 3B checkpoint and targeting
  its real 184,704-output `lm_head`, also completes the two-request CUDA
  pipeline smoke with the unchanged zero-delta result
  `semantic=12046,8433`, 128 latents, and 7,552 PCM samples in 2.11 seconds at
  1,113,972 KiB maximum host RSS. Loading the same adapter through the pure-C
  resident generator returns the same semantic IDs, 128 latents, and 3,776
  stereo frames while rechecking deterministic repetition, progress, and
  cancellation in 2.12 seconds at 1,114,088 KiB.
- The synthetic quantizer test (CTest `yue2-quantize-test`) builds a BF16
  GGUF with YuE2 tensor names and checks the Q4_K_M, Q8_0, and
  F16-from-Q4_K_M tensor plans. It also checks the streamed GGUF layout,
  byte-identical kept tensors, and preserved metadata and fingerprint. It
  covers `general.file_type`, the refusal of existing outputs, in-place writes,
  the VAE, and transcription GGUFs. On synthetic N(0, 0.02) weights, Q4_K
  cosine is 0.9976 and Q6_K is at least 0.9998. Quantized `get_rows` matches
  host dequantization to 1e-5. Q4_K and Q6_K `mul_mat` relative RMS against host
  math is 0.0101/0.0067 on the Windows CPU and 0.0089/0.0054 on the RTX 5070
  Laptop GPU; the backends quantize activations for these dot products.
- `yue2-quant-check` on the real transcription GGUFs (F32 reference, F16
  candidate) compares all 390 converted tensors with minimum cosine 1.000000,
  exercising the streamed checker on multi-gigabyte files.
- A current-source CUDA 12.8 rebuild on the RTX 5070 Laptop GPU passes the
  real-weight F16 transcription integration test (single window plus
  four-window stitching) in 1.5 seconds and the pure-C transcription test in
  1.0 second, with no quantization involved.
- Full conversion of the released 7,261,441,640-byte YuE2-3B source checkpoint
  into a 7,261,418,432-byte, 628-tensor BF16 GGUF and the released
  530,512,720-byte YuE2-VAE source into a 265,189,792-byte, 347-tensor F16
  GGUF. The latter folds 88 `weight_g`/`weight_v` pairs. The generated files'
  SHA-256 values are `d431a03f63beeda8493f6753db9ea3941c45d41b73a0b2fce05ae02cd5599349`
  and `a75b0124efe37451a2e01ae7d36e93a41e46b1ca1352c619cfa26ae645bc2e27`.
- The aarch64 C++ package inspector validates both real GGUFs, including
  architecture metadata, representative AR/NAR/VAE shapes, homogeneous
  BF16/F16 storage, folded weight norm, and embedded source hashes, in less
  than 0.01 seconds at 4,228 KiB maximum host RSS without allocating weights.
- The native text tokenizer matches all token IDs and decoded text from 10
  official tiktoken vectors spanning English contractions, multilingual
  scripts, decomposed Unicode, emoji/ZWJ sequences, ABC with CRLF, Unicode
  whitespace, and literal special-token-looking input. Four additional
  official fixtures cover the exact off/melody/full positive prompt and
  symbolic CFG negative-prefix contracts.
- On GB10 CUDA, the native 28-layer AR full-prefix graph matches the official
  float32 model over all 184,704 logits for a real 47-token request: maximum
  error 0.028267, RMS 0.002738, relative RMS 0.000857, with the same top token
  (ID 55). The run includes loading the 6.76 GiB BF16 GGUF and completes in
  8.66 seconds at 1,099,264 KiB maximum host RSS; weights reside primarily in
  accelerator memory.
- An official static-cache fixture checks both a 47-token prefill and the next
  single-token decode across all 184,704 logits. Native cached prefill has the
  same 0.000857 relative RMS as the cache-free graph; cached decode has maximum
  error 0.021933, RMS 0.002614, relative RMS 0.002307, and the same next token
  (ID 25). Four consecutive cache steps reproduce the official greedy ABC
  sequence `55, 25, 16, 198`, and a same-prefix two-cache CFG check exercises
  the guidance path.
- An official 32-step midpoint-flow fixture checks the complete 28-layer NAR
  branch on GB10 CUDA. Native 64-channel latents have max/mean/RMS error
  0.009071/0.001287/0.001903 and relative RMS 0.002532. Decoding native flow
  output with the native F16 VAE versus official flow output with the official
  FP32 VAE has max/mean/RMS waveform error
  0.001460/0.000347/0.000438.
- A real-weight complete-pipeline smoke test runs external ABC through semantic
  AR sampling, seeded NAR flow, and VAE decode twice on one loaded model pair.
  It reproduces codec IDs `12046, 8433`, all 128 latent values, and all 7,552
  interleaved PCM samples exactly across calls. The CUDA run completes in 6.29
  seconds at 1,125,552 KiB maximum host RSS, including model load and both
  requests.
- The pure-C shared-library ABI builds as `yue2.dll` on Windows and
  `libyue2.so` on aarch64 Linux. A GB10 real-weight test passes planar stereo
  PCM through the C transcription boundary and returns ABC, a 33-byte Standard
  MIDI file, and events JSON in 3.21 seconds at 631,512 KiB host RSS. A separate
  resident C generator test performs two identical complete requests, verifies
  byte-identical semantic IDs/latents/PCM, exercises every progress stage, and
  verifies cooperative cancellation without leaking a partial result.
- The native six-stage Oobleck VAE decoder matches the official FP32 decoder
  on the same deterministic two-frame latent fixture. With the 253 MiB F16
  GGUF, GB10 CUDA max/mean/RMS waveform error is
  0.001713/0.000363/0.000459 over 3,776 stereo samples; CPU error is
  0.000606/0.000129/0.000162. CUDA completes in 0.53 seconds at 424,836 KiB
  maximum host RSS.
- A 1,030-frame fixture verifies the tiling boundary: the tiled and full CUDA
  graphs have the same worst sample and essentially identical RMS. A
  2,050-frame, three-tile run produces 3,935,936 stereo samples in 2.88 seconds
  at 426,496 KiB host RSS, with mean/RMS error 0.000457/0.000964. Sparse
  long-sequence peaks reach 0.079 due to backend/storage accumulation rather
  than tile seams; the F32 VAE reduces the same run to 0.000750 RMS and a
  0.055 peak at roughly twice the weight size.
- Full conversion of the pinned 228,738,564-byte SheetSage2 adapter and
  2,529,812,848-byte MERT2 parent into a 1,355,653,664-byte GGUF containing
  1,039 tensors. The tested file SHA-256 is
  `29902ab4c24a5dd4571673210af4c93f9e70f3ad9fcd18dc5b923a839d4bcad0`.
- Native ConvNeXt output against the official reference: max absolute error
  0.0252385, mean 0.00347351, RMS 0.00443213.
- All 24 native Conformer blocks against a reference conditioned on the exact
  native subsampler output: max 0.00022459, mean 0.0000111943, RMS 0.0000147345.
- SheetSage2 learned mixture and 1024-to-512 projection: max 0.00107512, mean
  0.00018028, RMS 0.000230001.
- Six-layer BART decoder logits for a fixed memory/prefix fixture: max
  0.00337432, mean 0.000450494, RMS 0.000575422, cosine similarity 1.0.
- Grammar-constrained generation with the persistent self-attention KV cache
  and precomputed cross-attention K/V matches the reference's 13-token greedy
  sequence exactly.
- Real-checkpoint public API and CLI smoke tests on a deterministic 0.2-second
  WAV, including Standard MIDI output with no Python inference process.
- Melody-only and full-prompt CLI ABC outputs pass YuE2's strict native-dialect
  parser. A deterministic exporter stress fixture also passes with two complete
  measures, both melody voices, Vocal-only chords, explicit sounding pitches,
  a D-to-A-minor key change mirrored across voices, and a cross-bar tie.
- A synthetic downbeat fixture passes the same parser across consecutive 4/4,
  3/4, and 5/8 measures, including mirrored meter fields, the required shared
  unit length, structure group boundaries, an inline key change, and ties that
  continue across meter changes.
- A real-checkpoint four-window PCM smoke test covers right-context stopping,
  overlap-prefix reuse, accepted-region stitching, window provenance, and
  monotonic absolute event times. Release tests explicitly undefine `NDEBUG`,
  so assertion-based invariants remain active.
- A Windows CUDA 12.8 build targets native Blackwell `sm_120a` and runs on an
  RTX 5070 Laptop GPU without CPU fallback. The isolated CUDA ConvNeXt result
  has max/mean/RMS error 0.079464/0.007630/0.009920; after the complete encoder
  those errors fall to 0.001096/0.000050/0.000070, and projected memory remains
  at 0.006237/0.000964/0.001239. Decoder cosine similarity is 1.0 and the
  13-token constrained sequence is exact.
- A native aarch64 CUDA 13 Release build runs on the DGX Spark GB10 (compute
  capability 12.1) with the same GGUF checksum. Real-weight GB10 parity is:
  encoder max/mean/RMS 0.000711/0.000028/0.000041; projected memory
  0.002769/0.000571/0.000716; decoder max/mean/RMS
  0.030426/0.001298/0.001836 with cosine 1.0 and exact generated tokens.
- On GB10, the encoder/memory parity test takes 0.78 seconds at 444,476 KiB
  maximum host RSS, decoder/generation takes 0.84 seconds at 589,116 KiB, and
  the synthetic end-to-end/four-window API test takes 1.00 second at
  632,564 KiB.
- A full-prompt transcription of an existing 60.04-second YuE2 experiment
  output produces 295 events from 1,237 tokens in 4.20 seconds at 691,080 KiB
  maximum host RSS. YuE2's strict parser accepts the result as 37 synchronized
  measures with 294 instrumental notes and 45 Vocal-only chord annotations.
- The official SheetSage2 FP32 container run on the exact same PCM produces
  291 events and 1,217 tokens. Native CPU inference with the F32 GGUF also
  produces 291 events and 1,217 tokens; only one timestamp token differs
  (39.80 versus 39.84 seconds), and YuE2's exact score comparator reports an
  unqualified match for tempo, both voice meter grids, and every sounding note.
  This oracle path takes 146.30 seconds on the development CPU. CUDA/F16 is the
  intended interactive path, but long greedy sequences are not promised to be
  bit-identical across numerical backends.
- A 305-second real-audio loop exercises the released 300/200/100-second
  window/overlap/lookahead settings. A regression discovered here prevented
  the final window from being back-shifted: its start is now 100 seconds, its
  prefix is 2,171 rather than 3,872 tokens, and decoding reaches 304.81 seconds
  rather than truncating at 256.73. The two-window run takes 15.29 seconds at
  788,432 KiB maximum host RSS and its stitched 1,782-note ABC passes the same
  strict parser.

The reference fixture loader constructs MERT2 normally and loads the pinned
safetensors explicitly. This avoids a local Transformers low-memory-loading
compatibility issue that left the non-persistent rotary-frequency buffer
uninitialized and would otherwise make the oracle itself incorrect.

## Required before publishing quantized generation GGUFs

- Run `yue2-quant-check` on each real Q8_0/Q5_K_M/Q4_K_M file against the
  BF16 source.
- Compare quantized AR logits and flow latents with the official fixtures.
  The existing AR parity bound (relative RMS 0.003) is a BF16 gate; quantized
  tiers need their own recorded tolerances and top-token agreement.
- Run the complete generation smoke test and listening comparisons per tier.
- Measure peak VRAM per stage on an 8 GB GPU, including the F16 AR KV caches
  (about 0.11 MiB per token, two sessions under guidance).

## Required before calling transcription complete

- Extend the exact oracle comparison beyond the validated instrumental example
  to musically varied vocal and mixed arrangements, and compare a licensed
  non-repeated full-song result. Native long-window mechanics and production
  performance are already validated.
