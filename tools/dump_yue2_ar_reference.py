#!/usr/bin/env python3
"""Dump an official YuE2 full-prefix AR logit vector."""

from __future__ import annotations

import argparse
import struct
import sys
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--upstream", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--dtype", choices=("float32", "bfloat16"), default="float32")
    args = parser.parse_args()

    sys.path.insert(0, str(args.upstream / "src"))
    import torch
    from yue2.modeling_yue2 import StaticKVCache, YuE2ForCausalLM
    from yue2.protocol import SongRequest, token_prefixes
    from yue2.tokenization_yue2 import YuE2TextTokenizer

    tokenizer = YuE2TextTokenizer(args.model / "qwen.tiktoken")
    request = SongRequest(
        style="indie rock, warm vocal",
        lyrics="[Verse]\nCafé lights don't fade",
        cot="full",
    )
    prefix = token_prefixes(request, tokenizer)
    dtype = torch.float32 if args.dtype == "float32" else torch.bfloat16
    model = YuE2ForCausalLM.from_pretrained(
        args.model, torch_dtype=dtype, low_cpu_mem_usage=True).to("cuda").eval()
    cache = StaticKVCache(
        num_layers=model.config.num_hidden_layers,
        batch_size=1,
        num_kv_heads=model.config.num_key_value_heads,
        max_seq_len=len(prefix) + 4,
        head_dim=model.config.head_dim,
        dtype=dtype,
        device=torch.device("cuda"),
    )
    with torch.inference_mode():
        logits = model(
            input_ids=torch.tensor([prefix], device="cuda", dtype=torch.long),
            past_key_values=cache,
            use_cache=True,
            logits_to_keep=1,
        ).logits[0, -1].float().cpu().numpy()
        next_token = int(logits.argmax())
        next_logits = model(
            input_ids=torch.tensor([[next_token]], device="cuda", dtype=torch.long),
            past_key_values=cache,
            use_cache=True,
            logits_to_keep=1,
        ).logits[0, -1].float().cpu().numpy()

    greedy = [next_token]
    current = next_logits
    with torch.inference_mode():
        while len(greedy) < 4:
            token = int(current[:151643].argmax())
            greedy.append(token)
            if len(greedy) < 4:
                current = model(
                    input_ids=torch.tensor([[token]], device="cuda", dtype=torch.long),
                    past_key_values=cache,
                    use_cache=True,
                    logits_to_keep=1,
                ).logits[0, -1].float().cpu().numpy()

    output = bytearray(b"Y2AR\x03\x00\x00\x00")
    output += struct.pack("<I", len(prefix))
    output += struct.pack(f"<{len(prefix)}i", *prefix)
    output += struct.pack("<I", len(logits))
    output += logits.astype("<f4", copy=False).tobytes()
    output += struct.pack("<iI", next_token, len(next_logits))
    output += next_logits.astype("<f4", copy=False).tobytes()
    output += struct.pack("<I", len(greedy))
    output += struct.pack(f"<{len(greedy)}i", *greedy)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes(output)
    top = logits.argsort()[-5:][::-1]
    print(f"wrote {len(prefix)} tokens and two {len(logits)}-logit steps to {args.output}")
    print("top5", [(int(index), float(logits[index])) for index in top])
    next_top = next_logits.argsort()[-5:][::-1]
    print("next_token", next_token, "next_top5",
          [(int(index), float(next_logits[index])) for index in next_top])
    print("greedy_abc", greedy)


if __name__ == "__main__":
    main()
