# Distribution

How yue2.cpp model files are named, what each encoding contains, and how the
quantized tiers are produced. The convention is the one `sa3.cpp` and
`audiocraft.cpp` take from the GGUF specification (`ggml/docs/gguf.md`).

## Naming convention

`<BaseName>-<SizeLabel>-<Version>-<Encoding>[-<Type>].gguf`

- **BaseName**: `yue2`, `yue2-vae`, `sheetsage2-mert2`.
- **SizeLabel**: measured parameter class, on model-like components only. The
  VAE is exempt, as the autoencoders are in the sibling repositories.
- **Version**: `v1.0`. Bump it on any weight, conversion, or quantization
  recipe change.
- **Encoding**: `F32`, `F16`, `BF16`, `Q8_0`, `Q5_K_M`, `Q4_K_M`.
- **Type**: `LoRA` for adapters; omitted otherwise.

| File | Produced by | Notes |
|---|---|---|
| `yue2-3.6B-v1.0-BF16.gguf` | `convert_yue2_gguf.py` | Released precision; also `F16`, `F32` |
| `yue2-3.6B-v1.0-Q8_0.gguf` | `yue2-quantize --mix q8_0` | |
| `yue2-3.6B-v1.0-Q5_K_M.gguf` | `yue2-quantize --mix q5_k_m` | |
| `yue2-3.6B-v1.0-Q4_K_M.gguf` | `yue2-quantize --mix q4_k_m` | 8 GB GPU target |
| `yue2-vae-v1.0-F16.gguf` | `convert_yue2_gguf.py` | Also `F32`; never quantized |
| `sheetsage2-mert2-0.7B-v1.0-F16.gguf` | `convert_sheetsage2_gguf.py` | Also `F32`; never quantized |
| `<name>-v1.0-F16-LoRA.gguf` | `convert_yue2_lora.py` | Generation adapters |
| `yue2-semantic-tokenizer-<params>-v1.0-F16.gguf` | `convert_semantic_tokenizer_gguf.py` | Real-audio continuation tokenizer |

The YuE2-3B checkpoint holds 3,630.7M parameters, so its size label is `3.6B`
under the same helper the sibling converters use. The generation converter also
copies non-GGUF sidecars (`sidecars/yue2-qwen.tiktoken` and the JSON configs)
unchanged.

## Metadata

Converters stamp these through `tools/gguf_meta.py`. Loaders key off
`general.architecture` and `yue2.component`, never the file name.

| Key | Value |
|---|---|
| `general.basename` | e.g. `yue2` |
| `general.size_label` | e.g. `3.6B` (omitted for the VAE) |
| `general.version` | `v1.0` |
| `general.license` | `cc-by-nc-4.0` |
| `general.file_type` | GGUF file type (BF16 32, F16 1, Q8_0 7, Q4_K_M 15, Q5_K_M 17) |
| `general.base_model.N.*` | upstream name, organization, repository URL, optional revision |
| `yue2.checkpoint.sha256` | source checkpoint SHA-256; LoRA adapters bind to it |
| `yue2.quantization.encoding` | set by `yue2-quantize`, e.g. `Q4_K_M` |
| `yue2.quantization.version` | tensor-plan recipe version, currently `1` |

GGUFs converted before this convention lack the `general.*` catalog keys but
load unchanged. Reconvert them before publication.

## Quantization recipe (version 1)

| Tensor class | Q4_K_M | Q5_K_M | Q8_0 |
|---|---|---|---|
| AR/NAR `q/k/o_proj`, `gate/up_proj` | Q4_K | Q5_K | Q8_0 |
| AR/NAR `v_proj`, `down_proj` | Q6_K | Q6_K | Q8_0 |
| `model.embed_tokens.weight`, `lm_head.weight` | Q6_K | Q6_K | Q8_0 |
| Norms, biases (1-D) | source | source | source |
| `latent_pos_embed.pe`, `vae2llm.*`, `llm2vae.*`, `time_embedder.*` | source | source | source |

- The promotion rule follows `sa3-quantize` and llama.cpp's `*_K_M` mixes.
- A matrix whose row width is not a whole number of target blocks keeps its
  source storage. That can't happen for the released YuE2 widths (2048 and
  6144).
- The flow boundary stays at source precision for two reasons. These tensors sit
  directly on the continuous latent path and total only about 55M parameters.
  Also, the latent position table is read through a view and an F32 cast, which
  quantized storage cannot serve.
- The `f16` and `f32` mixes re-encode every 2-D matrix and dequantize a
  quantized input.

## Memory budget on an 8 GB GPU (estimates)

| Component | Size |
|---|---|
| Main model BF16 / Q8_0 / Q4_K_M | 6.76 GiB / ~3.9 GiB / ~2.4 GiB |
| VAE F16 | 253 MiB |
| AR KV cache, F16 | ~0.11 MiB per token per session |
| Semantic phase at 9,000 tokens | ~1.1 GiB, doubled under guidance |

These are arithmetic estimates, not measurements. KV capacity can matter as much
as weights, so stage-level VRAM measurement is required before the Q4_K_M tier
is declared to fit.

## Producing the tiers

Quantized files are produced on the DGX Spark, where the BF16 source lives. The
quantizer itself is covered by `yue2-quantize-test` on every platform.

```bash
cmake -S . -B build -DYUE2_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=native
cmake --build build --config Release -j
ctest --test-dir build -C Release --output-on-failure

python tools/convert_yue2_gguf.py --model models/YuE2-3B --vae models/YuE2-Vae \
  --out models/YuE2-3B-GGUF
for mix in q8_0 q5_k_m q4_k_m; do
  build/bin/yue2-quantize --in models/YuE2-3B-GGUF/yue2-3.6B-v1.0-BF16.gguf --mix $mix
done
for tier in Q8_0 Q5_K_M Q4_K_M; do
  build/bin/yue2-quant-check --ref models/YuE2-3B-GGUF/yue2-3.6B-v1.0-BF16.gguf \
    --quant models/YuE2-3B-GGUF/yue2-3.6B-v1.0-$tier.gguf
  build/bin/yue2-inspect-generation models/YuE2-3B-GGUF/yue2-3.6B-v1.0-$tier.gguf \
    models/YuE2-3B-GGUF/yue2-vae-v1.0-F16.gguf
done
```

Validation still required before publication is listed in
[validation.md](validation.md).

## Hugging Face repository and download profiles

All components and encodings live in one model-family repository:
[`thepatch/YuE2-3B-GGUF`](https://huggingface.co/thepatch/YuE2-3B-GGUF).
This follows the sa3.cpp convention: a quantization tier is a file selection,
not a separate repository.

`models.sh`, `models.cmd`, and `tools/download_models.py` resolve the same three
profiles from `tools/model_artifacts.py`:

| profile | files |
|---|---|
| `core` | generation model, VAE, Qwen tokenizer |
| `transcribe` | core plus SheetSage2/MERT2 |
| `full` | transcribe plus the instrumental adapter and matched real-audio tokenizer/NAR pair |

The published default is `full` at Q4_K_M. A requested encoding is accepted by
the downloaders only after that tier has actually been published, preventing a
documented command from resolving to missing files.
