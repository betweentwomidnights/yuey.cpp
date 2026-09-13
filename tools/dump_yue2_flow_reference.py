#!/usr/bin/env python3
"""Dump an official YuE2 NAR midpoint-flow fixture with explicit noise."""

from __future__ import annotations

import argparse
import struct
import sys
from pathlib import Path


def ids(values) -> bytes:
    values = list(values)
    return struct.pack("<I", len(values)) + struct.pack(f"<{len(values)}i", *values)


def floats(values) -> bytes:
    values = values.reshape(-1).float().cpu().numpy().astype("<f4", copy=False)
    return struct.pack("<I", len(values)) + values.tobytes()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--upstream", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--vae", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=1)
    args = parser.parse_args()
    sys.path.insert(0, str(args.upstream / "src"))

    import torch
    from yue2.modeling_yue2 import YuE2ForCausalLM
    from yue2.modeling_vae import YuE2VAE
    from yue2.nar import CachedNAR, Chunk
    from yue2.protocol import CODEC_OFFSET, MUSIC_END, SongRequest, token_prefixes
    from yue2.tokenization_yue2 import YuE2TextTokenizer

    tokenizer = YuE2TextTokenizer(args.model / "qwen.tiktoken")
    request = SongRequest(
        style="acoustic, intimate",
        lyrics="[Verse]\nA quiet line",
        cot="full",
        abc="X:1\nM:4/4\nK:C\nC2 E2 G4|",
    )
    abc_ids = tokenizer.encode(request.abc)
    prefix = token_prefixes(request, tokenizer, abc_ids)
    codec = [0, 1234]
    noise = torch.linspace(-0.75, 0.875, len(codec) * 64, dtype=torch.float32).reshape(len(codec), 64)
    model = YuE2ForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.float32, low_cpu_mem_usage=True).to("cuda").eval()
    chunk = Chunk(
        prefix + [CODEC_OFFSET + value for value in codec] + [MUSIC_END],
        noise,
    )
    engine = CachedNAR(model, chunk)
    try:
        result = engine.solve(args.steps)
    finally:
        engine.close()
    vae = YuE2VAE.from_pretrained(args.vae, decoder_only=True, device="cuda")
    with torch.inference_mode():
        planar = vae.decode(result.transpose(0, 1).unsqueeze(0).contiguous().to("cuda"))
    interleaved = planar[0].transpose(0, 1).contiguous().cpu()

    output = bytearray(b"Y2FL\x02\x00\x00\x00")
    output += struct.pack("<I", args.steps)
    output += ids(prefix)
    output += ids(codec)
    output += floats(noise)
    output += floats(result)
    output += floats(interleaved)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes(output)
    print(f"wrote {len(codec)} frames, {args.steps} midpoint steps to {args.output}")
    print("result", result.reshape(-1)[:8].tolist())
    print("audio_samples", len(interleaved), "range", interleaved.min().item(), interleaved.max().item())


if __name__ == "__main__":
    main()
