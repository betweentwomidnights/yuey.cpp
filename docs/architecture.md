# Architecture

## Product boundary

YuE2 generation and audio transcription are related but separate model paths.
The upstream cover workflow does not feed audio into YuE2. It first transcribes
audio with SheetSage2/MERT2, then supplies the resulting ABC score to YuE2.

The reusable C++ API therefore starts with `Transcriber`, not a generation
session. This boundary is suitable for a CLI today and gary4local/gary4juce
embedding later:

```text
TranscriptionOptions + PCM/audio path
                  |
                  v
          yue2::Transcriber (WAV path or mono PCM buffer)
                  |
                  +-- tokens (lossless SheetSage2 v1 sequence)
                  +-- events (stable structured representation)
                  +-- ABC (YuE2 conditioning artifact)
                  +-- MIDI (interchange/playback artifact)
                  +-- warnings and timing/provenance
```

## Transcription graph

1. Decode input, mix to mono, and resample to 24 kHz without amplitude
   normalization.
2. Pad each inference window to the configured duration (300 seconds by
   default, matching the released model).
3. Produce 128-bin power log-mel features (`n_fft=2048`, `hop=240`).
4. Run MERT2-FS: three ConvNeXt subsampling blocks and 24 Conformer blocks.
5. Apply SheetSage2's learned 25-way layer mixture and 1024-to-512 projection.
6. Greedily decode the six-layer BART decoder with the published event grammar,
   persistent self-attention KV state, and precomputed cross-attention K/V.
7. Decode typed events, accept only each window's non-lookahead region,
   re-encode overlap history as grammar-valid context, preserve global subbeat
   continuity, and export ABC/MIDI.

MERT2 attention adapters are merged during GGUF conversion. The GGUF stores
the merged encoder plus SheetSage2 head as one independently loadable
transcription checkpoint. This mirrors the official inference path, which
merges rank-64 attention LoRAs into MERT2 in float32 before inference.

## Generation package boundary

Generation uses two independently loadable GGUFs plus tokenizer/config
sidecars:

```text
YuE2-3B BF16 GGUF (AR + NAR/flow, native Hugging Face tensor names)
        +
YuE2-VAE F16 GGUF (Oobleck encoder/decoder, weight norm folded)
        +
Qwen tiktoken + model/generation/VAE JSON
```

`inspect_generation_package()` reads only metadata and tensor descriptors. It
validates the released 28-layer, 2048-wide, 16/8-head architecture, the
24,576-token/latent context, representative AR/NAR tensors, the 48 kHz stereo
VAE geometry, and the absence of residual `weight_g`/`weight_v` tensors. This
keeps malformed or mismatched packages from reaching backend allocation.

The first executable generation component is `VaeDecoder`. It consumes
time-major 64-channel latents, explicitly converts the host layout to GGML's
convolution layout, runs the released six Oobleck upsampling blocks, and
returns interleaved stereo PCM. Inputs longer than 1024 latent frames use
16-frame left/right halos and copy exact output cores into one natural-length
waveform; there is no crossfade. This makes activation memory proportional to
tile size rather than song duration.

The text boundary is implemented independently of locale or platform Unicode
libraries. `TextTokenizer` performs strict UTF-8 decoding, Unicode 15 NFC,
Qwen's Unicode-aware pre-tokenization, and raw-byte ranked BPE. It deliberately
uses ordinary encoding so protocol-looking user text stays user text. Prompt
helpers mirror upstream `protocol.py` and retain exact positive-branch ABC IDs
for symbolic classifier-free guidance.

`AutoregressiveModel` validates and loads the released combined GGUF through
the same backend seam as transcription and the VAE. Its exact-prefix
graph implements the 28-layer AR branch: RMSNorm, separate Q/K/V projections,
per-head Q/K normalization, NeoX RoPE, grouped-query causal attention, SwiGLU,
final normalization, and the untied LM head. Bounded request-local sessions
store per-layer K/V in F16 and support both whole-prompt prefill and incremental
token appends. The native loop implements the released ABC and semantic
vocabulary windows, minimum length, occurrence-counted windowed repetition
penalty, temperature, top-k/top-p filtering, stop tokens, and two-cache
classifier-free guidance.

Generation LoRAs use a separate `yue2_lora` GGUF. Each adapter buffer is
allocated on the same backend as the base model and bound by base-weight
pointer after strict metadata, fingerprint, target, rank, dtype, and shape
validation. Every eligible AR/NAR linear node evaluates the adapter
functionally as `W*x + scale*B*(A*x)`. Base tensors remain immutable, avoiding
both merge cost and a second quantization pass. Multiple plain LoRAs compose
additively, and strength zero is an explicit graph-level bypass.

Quantized generation packages come from `yue2-quantize`, which rewrites only
the main GGUF's 2-D projection and embedding matrices. Every AR/NAR
projection already reaches GGML through `mul_mat` and the embedding through
`get_rows`, both of which accept K-quant storage on CPU and CUDA, so the loader
and graphs need no quantization-specific path. The latent position table is
read through a view and an F32 cast, so it is excluded along with the rest of
the flow boundary. The VAE and the transcription model are not quantized.

The same model owns the released 28-layer NAR branch. Flow execution prefills
the AR prefix and semantic codec sequence, projects seeded 64-channel noise,
adds timestep and latent-position embeddings, attends over the cached AR keys
and values, and integrates velocities with the upstream midpoint solver. Long
semantic streams use the released context-derived chunk formula. The
explicit-noise API is the deterministic parity boundary; the seeded overload
uses the native standard C++ random stream. `GenerationPipeline` composes
optional ABC planning, semantic sampling, flow synthesis, and tiled VAE decode
while keeping both GGUFs resident across calls. Model state is shared safely by
sessions while each mutable GGML scheduler serializes graph submissions.

## GGML ownership

`ggml/` points at `betweentwomidnights/ggml`, the same fork and revision used
by the sibling music projects. Model-specific tensor code lives in this
repository; fixes that are genuinely general belong in the shared submodule.
The engine must not vendor or link a second GGML copy.

## Milestones

1. Contract, token grammar, checkpoint conversion, and deterministic parity
   fixtures.
2. 24 kHz WAV front end and MERT2 feature/subsampling parity.
3. Conformer encoder, learned layer mixture, and projection parity.
4. KV-cached BART decoder, grammar-constrained generation, typed metadata/event
   decoding, timestamp interpolation, YuE2-parser-validated native ABC, and
   two-voice MIDI export. (Complete.)
5. Long-song overlap-prefix stitching, including fixed-hop partial-tail
   windows that cannot inflate the decoder prefix. (Implemented and exercised
   on a 305-second two-window DGX Spark run.)
6. Native downbeat/meter reconstruction, CUDA execution, and an exact
   score-level comparison with the official FP32 implementation. (Complete.)
7. YuE2 AR/NAR/flow/VAE generation. Native checkpoint conversion,
   metadata-only package validation, the exact tokenizer/prompt contract, the
   KV-cached AR sampling graph, midpoint NAR/flow latent synthesis, tiled VAE
   decoder, reusable complete pipeline, and WAV CLI are complete.
8. Library/server bindings for generation and gary4local/gary4juce integration.
   The stable pure-C shared-library ABI, resident split contexts, in-memory
   planar/interleaved transcription, result ownership, progress, and cooperative
   cancellation are complete. `yue2-server` provides the gary4local-style
   async HTTP job transport for generation, covers, and transcription; wiring
   it into gary4local and a gary4juce panel remains.
9. Adapter support. PEFT-to-GGUF conversion and functional AR/NAR LoRA
   inference are complete. Native training and DoRA remain, using the shared
   GGML optimizer path after training-target and gradient parity validation.
