#!/usr/bin/env python3
"""Create deterministic official YuE2-VAE decoder parity fixtures."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--upstream", type=Path, required=True, help="upstream YuE checkout")
    parser.add_argument("--vae", type=Path, required=True, help="released YuE2-VAE directory")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--frames", type=int, default=2)
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--round-effective-weights-f16",
        action="store_true",
        help="fold weight norm, round every parameter through f16, then run f32",
    )
    args = parser.parse_args()
    if args.frames <= 0:
        raise SystemExit("--frames must be positive")

    sys.path.insert(0, str(args.upstream.resolve() / "src"))
    from yue2.modeling_vae import YuE2VAE

    args.out.mkdir(parents=True, exist_ok=True)
    indices = torch.arange(64 * args.frames, dtype=torch.float32)
    latent_time_major = (
        torch.sin((indices + 1.0) * 0.017) +
        0.1 * torch.cos((indices + 3.0) * 0.031)
    ).reshape(args.frames, 64).contiguous()
    latent = latent_time_major.transpose(0, 1).unsqueeze(0).contiguous()

    model = YuE2VAE.from_pretrained(
        args.vae.resolve(), decoder_only=True, device=args.device
    )
    folded = 0
    if args.round_effective_weights_f16:
        from torch.nn.utils import remove_weight_norm

        for module in model.modules():
            try:
                remove_weight_norm(module)
                folded += 1
            except ValueError:
                pass
        with torch.no_grad():
            for parameter in model.parameters():
                parameter.copy_(parameter.half().float())
    with torch.inference_mode():
        planar = model.decode(latent).cpu()
    interleaved = planar[0].transpose(0, 1).contiguous()
    latent_time_major.numpy().tofile(args.out / "latents.f32")
    interleaved.numpy().tofile(args.out / "expected-interleaved.f32")
    print(
        f"frames={args.frames} samples={interleaved.shape[0]} "
        f"min={interleaved.min().item():.9g} max={interleaved.max().item():.9g} "
        f"folded={folded}"
    )


if __name__ == "__main__":
    main()
