#!/usr/bin/env python3
"""Offline posterior-mean VAE targets from the existing folded F16/F32 GGUF.

This small CPU/PyTorch reference enables the VAE-only reconstruction control
without downloading another checkpoint. It is preprocessing, not GGML training.
Requires torch, gguf, numpy, soundfile, scipy. Latents are raw little-endian F32
[frames,64], one frame per 1920 input samples (25 Hz at 48 kHz).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import gguf
import numpy as np
import torch
import torch.nn.functional as F

from yue2_training_audit import digest, load_audio


class Encoder:
    def __init__(self, path):
        self.reader = gguf.GGUFReader(str(path))
        self.weights = {}
        for tensor in self.reader.tensors:
            if not tensor.name.startswith("encoder."):
                continue
            if tensor.tensor_type not in (gguf.GGMLQuantizationType.F16, gguf.GGMLQuantizationType.F32):
                raise ValueError("VAE encoder requires folded F16/F32 weights")
            self.weights[tensor.name] = torch.from_numpy(tensor.data.astype(np.float32, copy=True))

    def conv(self, x, name, stride=1, dilation=1, padding=None):
        w, b = self.weights[name + ".weight"], self.weights[name + ".bias"]
        if padding is None:
            padding = (w.shape[-1]-1)*dilation//2
        return F.conv1d(x, w, b, stride=stride, dilation=dilation, padding=padding)

    def snake(self, x, name):
        alpha = self.weights[name + ".alpha"].exp().reshape(1, -1, 1)
        beta = self.weights[name + ".beta"].exp().reshape(1, -1, 1)
        return x + (alpha*x).sin().square() / (beta + 1e-9)

    @torch.inference_mode()
    def encode(self, audio):
        x = torch.from_numpy(np.ascontiguousarray(audio.T)).unsqueeze(0)
        if x.shape[1] == 1:
            x = x.repeat(1, 2, 1)
        if x.shape[1] != 2 or x.shape[-1] < 1920:
            raise ValueError("expected mono/stereo audio of at least one latent frame")
        x = self.conv(x, "encoder.layers.0")
        for layer, stride in enumerate((2, 2, 4, 4, 5, 6), 1):
            name = f"encoder.layers.{layer}.layers"
            for unit, dilation in enumerate((1, 3, 9)):
                prefix = f"{name}.{unit}.layers"
                y = self.conv(self.snake(x, prefix + ".0"), prefix + ".1", dilation=dilation)
                y = self.conv(self.snake(y, prefix + ".2"), prefix + ".3")
                x = x + y
            x = self.conv(self.snake(x, name + ".3"), name + ".4",
                          stride=stride, padding=(stride+1)//2)
        x = self.conv(self.snake(x, "encoder.layers.7"), "encoder.layers.8")
        if x.shape[1] != 128:
            raise ValueError("unexpected VAE posterior shape")
        # The other 64 channels parameterize posterior scale. Training targets
        # and the reconstruction control use the deterministic mean.
        result = x[0, :64].T.contiguous().numpy().astype("<f4")
        if not np.isfinite(result).all():
            raise ValueError("nonfinite posterior mean")
        return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vae", type=Path, required=True)
    parser.add_argument("--audio", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    if args.out.exists() or args.out.with_suffix(".json").exists():
        parser.error("output exists; choose a new path")
    torch.set_num_threads(args.threads)
    latents = Encoder(args.vae).encode(load_audio(args.audio, 48000))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    latents.tofile(args.out)
    metadata = {"shape": list(latents.shape), "layout": "time-major", "dtype": "little-endian-f32",
                "posterior": "mean", "vae_sha256": digest(args.vae), "audio_sha256": digest(args.audio),
                "audio": str(args.audio.resolve()), "vae": str(args.vae.resolve())}
    args.out.with_suffix(".json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
