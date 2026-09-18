# Instrumental and real-audio adapters

YuE2's optional Mothersuperior adapter pair adds two distinct capabilities:

- the instrumental AR adapter changes score-first planning and semantic
  generation so the instrumental toggle is materially more reliable; and
- the real-audio package maps source audio into YuE2 semantic codec IDs and
  supplies the matching NAR adapter needed to reconstruct those IDs.

The weights are not bundled. Their upstream license is CC BY-NC 4.0; downstream
products must evaluate that license independently before distributing or
hosting them. yue2.cpp only supplies generic conversion and runtime support.

## Convert the adapters

The released safetensors omit PEFT's usual `.weight` suffix and the NAR package
contains four full replacement tensors in addition to rank-32 LoRA factors.
`convert_yue2_lora.py` handles both layouts directly:

```bash
python tools/convert_yue2_lora.py \
  --input YuE2-instrumental-cot-full-loras/ar_lora_inst_v3abc.bf16.safetensors \
  --type f16 \
  --output models/yue2-instrumental-cot-full-v1.0-F16-LoRA.gguf

python tools/convert_yue2_lora.py \
  --input yue2-mothersuperior-realaudio-tokenizer-v4/nar_lora_joint_v9.bf16.safetensors \
  --type f16 \
  --output models/yue2-realaudio-nar-v9-v1.0-F16-LoRA.gguf
```

The converter defaults to `alpha = rank`, matching the released adapters' scale
1.0 rule. `--base-sha256` can bind a converted adapter to one generation GGUF.
The two released adapters can be loaded together without a runtime phase switch:
the instrumental file targets the AR `self_attn`/`mlp` tensors, while the v9
file targets `nar_self_attn`/`nar_mlp` and the four NAR boundary replacements.
Do not assume that separation for unrelated third-party adapters; inspect their
target names before assigning them an automatic server role.

## Convert the semantic tokenizer

Download the unmodified `m-a-p/MERT-v2-FullSong` snapshot and the v9 tokenizer
head. Do not reuse the SheetSage2 GGUF: its attention adapter is merged into
MERT, while this tokenizer was trained against the unmodified parent.

```bash
python tools/convert_semantic_tokenizer_gguf.py \
  --mert models/MERT-v2-FullSong \
  --head yue2-mothersuperior-realaudio-tokenizer-v4/tokenizer_head_joint_v9.bf16.safetensors \
  --out models
```

The resulting GGUF contains MERT layer-20 plus the eight-layer, 25 Hz,
32,768-class tokenizer head. It is loaded only during `/continue` and released
before the 3B generation model is loaded.

## Start the service

```bash
yue2-server --models-dir models --encoding Q4_K_M \
  --instrumental-lora yue2-instrumental-cot-full \
  --continuation-lora yue2-realaudio-nar-v9
```

`/continue` requires the continuation adapter. This is deliberate: semantic
tokens produced by the real-audio head should not be rendered through unrelated
stock NAR weights. The server transcribes the input for an ABC planning prefix,
extracts its audio-semantic prefix, extends both, and returns one reconstructed
source-plus-continuation WAV.
