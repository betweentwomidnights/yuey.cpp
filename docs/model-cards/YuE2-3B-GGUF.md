---
language:
- en
license: cc-by-nc-4.0
pipeline_tag: text-to-audio
base_model:
- m-a-p/YuE2-3B
- m-a-p/YuE2-Vae
- m-a-p/SheetSage2
base_model_relation: quantized
tags:
- audio-generation
- music
- transcription
- gguf
- yue2
- yuey.cpp
---

# YuE2 3B — GGUF for yuey.cpp

Native GGUF conversions and optional adapters for
[yuey.cpp](https://github.com/betweentwomidnights/yuey.cpp), a C++/GGML YuE2
runtime. It supports music generation, SheetSage2 transcription and MIDI export,
score-conditioned remixing, instrumental planning, and real-audio continuation.

## Download one complete set

The repository is one multi-file model family. Choose BF16 for reference
precision, Q8_0 for a high-quality quantized model, or Q4_K_M for an 8 GB-class
laptop GPU. Q4_K_M remains the downloader default.

| encoding | generation model | `core` | `transcribe` | `full` |
|---|---:|---:|---:|---:|
| BF16 | 6.76 GiB | 7.01 GiB | 8.27 GiB | 9.73 GiB |
| Q8_0 | 3.64 GiB | 3.89 GiB | 5.15 GiB | 6.61 GiB |
| Q4_K_M | 2.36 GiB | 2.60 GiB | 3.87 GiB | 5.32 GiB |

`core` contains generation, the F16 VAE, and the text tokenizer. `transcribe`
adds F16 SheetSage2/MERT2. `full` adds the instrumental and real-audio adapters
plus the semantic tokenizer. Runtime memory also includes activations and
backend overhead, so model download size is not a VRAM requirement.

Use the repository downloader rather than selecting files manually:

```bash
git clone --recurse-submodules https://github.com/betweentwomidnights/yuey.cpp.git
cd yuey.cpp
./models.sh --profile full
# Windows: models.cmd --profile full

# Higher-precision alternatives:
./models.sh --encoding q8_0 --profile full
./models.sh --encoding bf16 --profile full
```

The faster Python path uses `huggingface_hub` and `hf_xet`:

```bash
python -m pip install -U "huggingface_hub[hf_xet]"
python tools/download_models.py --profile full
```

## Files

| file | role |
|---|---|
| `yue2-3.6B-v1.0-BF16.gguf` | reference-precision YuE2 AR, NAR/flow generation model |
| `yue2-3.6B-v1.0-Q8_0.gguf` | high-quality quantized generation model |
| `yue2-3.6B-v1.0-Q4_K_M.gguf` | laptop-oriented quantized generation model |
| `yue2-vae-v1.0-F16.gguf` | audio VAE decoder |
| `yue2-qwen.tiktoken` | text and score tokenizer |
| `sheetsage2-mert2-0.7B-v1.0-F16.gguf` | audio-to-score transcription and MIDI |
| `yue2-instrumental-cot-full-v1.0-F16-LoRA.gguf` | optional instrumental AR adapter |
| `yue2-realaudio-nar-v9-v1.0-F16-LoRA.gguf` | NAR adapter paired with the semantic tokenizer |
| `yue2-semantic-tokenizer-0.7B-v1.0-F16.gguf` | 25 Hz real-audio continuation tokenizer |

The real-audio NAR adapter and semantic tokenizer are a matched pair. Do not use
the semantic tokens with the stock NAR weights. The instrumental adapter can be
bypassed per request for controlled comparisons.

## Sources and attribution

- Generation and VAE: [m-a-p/YuE2-3B](https://huggingface.co/m-a-p/YuE2-3B)
  and [m-a-p/YuE2-Vae](https://huggingface.co/m-a-p/YuE2-Vae).
- Transcription: [m-a-p/SheetSage2](https://huggingface.co/m-a-p/SheetSage2)
  with [m-a-p/MERT-v2-FullSong](https://huggingface.co/m-a-p/MERT-v2-FullSong).
- Instrumental adapter:
  [Mothersuperior/YuE2-instrumental-cot-full-loras](https://huggingface.co/Mothersuperior/YuE2-instrumental-cot-full-loras).
- Real-audio head and NAR adapter:
  [Mothersuperior/yue2-mothersuperior-realaudio-tokenizer-v4](https://huggingface.co/Mothersuperior/yue2-mothersuperior-realaudio-tokenizer-v4).

These are format conversions and quantizations for inference; yuey.cpp did not
train the upstream checkpoints. See `SHA256SUMS` for release-file checksums.

## License

The converted weights and adapters are distributed under their upstream
**Creative Commons Attribution-NonCommercial 4.0** terms. They are for
non-commercial use unless the relevant rights holders grant separate permission.
The yuey.cpp runtime itself is MIT licensed.
