#!/usr/bin/env python3
"""Dump a deterministic official-MERT2 ConvNeXt reference for native parity tests."""

from __future__ import annotations

import argparse
import math
import sys
import types
from pathlib import Path

import numpy as np
import torch
from safetensors import safe_open


def waveform(sample_count: int) -> torch.Tensor:
    index = torch.arange(sample_count, dtype=torch.float64)
    signal = (
        0.10 * torch.sin(2.0 * math.pi * 311.0 * index / 24000.0)
        + 0.05 * torch.cos(2.0 * math.pi * 997.0 * index / 24000.0)
        + 0.002 * ((index.remainder(97.0) - 48.0) / 48.0)
    )
    return signal.float().unsqueeze(0)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--adapter", type=Path, help="SheetSage2 directory whose attention LoRAs are merged")
    parser.add_argument("--gguf-f16", action="store_true", help="round matrix weights like the default GGUF")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--samples", type=int, default=4800)
    parser.add_argument("--full", action="store_true", help="dump all 24 Conformer blocks, not just subsampling")
    parser.add_argument("--layers", type=int, help="dump the selected 1-based Conformer depth")
    parser.add_argument("--memory", action="store_true", help="dump SheetSage2's mixed 512-wide encoder memory")
    parser.add_argument("--input-hidden", type=Path, help="start the Conformer stack from a raw [T,1024] f32 fixture")
    parser.add_argument("--stage", type=int, choices=range(1, 7), help="dump a selected layer-zero residual/query stage")
    parser.add_argument("--zero-positions", action="store_true", help="use identity rotary positions for attention diagnosis")
    args = parser.parse_args()

    # The downloaded model directory contains relative imports but its folder
    # name is not a valid Python identifier. Mount it as a temporary package.
    package = types.ModuleType("mert2_reference")
    package.__path__ = [str(args.model)]
    sys.modules[package.__name__] = package
    from mert2_reference.configuration_mert2 import MERT2Config  # type: ignore
    from mert2_reference.modeling_mert2 import ConvNextBlock, MERT2MelFrontend, MERT2Model  # type: ignore

    config = MERT2Config.from_pretrained(args.model)
    if args.full or args.layers or args.memory or args.stage:
        config._attn_implementation = "sdpa"
        # Instantiate normally before loading. Transformers' low-memory loader
        # can leave MERT2's non-persistent inv_freq buffer uninitialized because
        # it is intentionally absent from the checkpoint state dict.
        model = MERT2Model(config).eval()
        with safe_open(args.model / "model.safetensors", framework="pt", device="cpu") as source:
            state = {name: source.get_tensor(name) for name in source.keys()}
        incompatible = model.load_state_dict(state, strict=False)
        del state
        missing, unexpected = incompatible.missing_keys, incompatible.unexpected_keys
        if missing or unexpected:
            raise ValueError(f"incomplete MERT2 load: missing={missing}, unexpected={unexpected}")
        layer_weight = projection_weight = projection_bias = None
        if args.adapter:
            with safe_open(args.adapter / "model.safetensors", framework="pt", device="cpu") as weights:
                layer_weight = weights.get_tensor("layer_weight")
                projection_weight = weights.get_tensor("encoder_projection.weight")
                projection_bias = weights.get_tensor("encoder_projection.bias")
                with torch.no_grad():
                    for layer_index, layer in enumerate(model.layers):
                        for projection in ("query_proj", "key_proj", "value_proj", "out_proj"):
                            prefix = f"adapter.layers.{layer_index}.attn.{projection}"
                            update = weights.get_tensor(prefix + ".lora_B.weight") @ weights.get_tensor(
                                prefix + ".lora_A.weight"
                            )
                            getattr(layer.attn, projection).weight.add_(update, alpha=2.0)
        if args.gguf_f16:
            with torch.no_grad():
                for name, parameter in model.named_parameters():
                    if name.startswith(("subsampling_module.", "layers.")) and parameter.ndim > 1:
                        parameter.copy_(parameter.half().float())
        with torch.inference_mode():
            if args.input_hidden:
                raw = np.fromfile(args.input_hidden, dtype=np.float32)
                hidden_input = torch.from_numpy(raw.reshape(1, -1, 1024).copy())
            else:
                hidden_input = model.subsampling_module(model.feature_extractor(waveform(args.samples)))
            if args.stage:
                layer = model.layers[0]
                positions = model.embed_positions(hidden_input)
                if args.zero_positions:
                    positions = (torch.ones_like(positions[0]), torch.zeros_like(positions[1]))
                selected = hidden_input + 0.5 * layer.ffn1(layer.ffn1_layer_norm(hidden_input))
                if args.stage == 6:
                    attention_input = layer.attn_layer_norm(selected)
                    shape = (1, attention_input.shape[1], layer.attn.num_heads, layer.attn.head_dim)
                    query = layer.attn.query_proj(attention_input).reshape(shape)
                    selected = query * positions[0] + torch.cat((-query[..., 32:], query[..., :32]), -1) * positions[1]
                elif args.stage >= 2:
                    attention_input = layer.attn_layer_norm(selected)
                    if args.stage == 2:
                        shape = (1, attention_input.shape[1], layer.attn.num_heads, layer.attn.head_dim)
                        query = layer.attn.query_proj(attention_input).reshape(shape)
                        key = layer.attn.key_proj(attention_input).reshape(shape)
                        query = query * positions[0] + torch.cat((-query[..., 32:], query[..., :32]), -1) * positions[1]
                        key = key * positions[0] + torch.cat((-key[..., 32:], key[..., :32]), -1) * positions[1]
                        score = torch.einsum("bthd,bshd->bhts", query, key) * 0.125
                        print(
                            f"attention score range=[{score.min().item():.9g}, {score.max().item():.9g}], "
                            f"qmax={query.abs().max().item():.9g}, kmax={key.abs().max().item():.9g}"
                        )
                    selected = selected + layer.attn(attention_input, positions)
                if 3 <= args.stage < 6:
                    selected = selected + layer.conv_module(selected)
                if 4 <= args.stage < 6:
                    selected = selected + 0.5 * layer.ffn2(layer.ffn2_layer_norm(selected))
                if args.stage == 5:
                    selected = layer.final_layer_norm(selected)
            elif args.memory:
                if layer_weight is None:
                    raise ValueError("--memory requires --adapter")
                hidden = hidden_input
                positions = model.embed_positions(hidden)
                mixture = hidden * torch.softmax(layer_weight, dim=0)[0]
                for weight, layer in zip(torch.softmax(layer_weight, dim=0)[1:], model.layers):
                    hidden = layer(hidden, positions)
                    mixture = mixture + hidden * weight
                if args.gguf_f16:
                    projection_weight = projection_weight.half().float()
                selected = torch.nn.functional.linear(mixture, projection_weight, projection_bias)
            else:
                if args.input_hidden:
                    hidden = hidden_input
                    positions = model.embed_positions(hidden)
                    states = []
                    for layer in model.layers:
                        hidden = layer(hidden, positions)
                        states.append(hidden)
                    selected = states[args.layers - 1] if args.layers else hidden
                else:
                    output = model(waveform(args.samples), output_hidden_states=bool(args.layers))
                    selected = output.hidden_states[args.layers - 1] if args.layers else output.last_hidden_state
            result = selected.float().contiguous().cpu().numpy()
    else:
        frontend = MERT2MelFrontend(config)
        channels = [config.num_mel_bins, *config.subsampling_channels]
        subsampling = torch.nn.Sequential(
            *[
                ConvNextBlock(
                    channels[index],
                    channels[index + 1],
                    (1, 2, 2)[index],
                    config.subsampling_depths[index],
                    config.subsampling_layer_norm_eps,
                )
                for index in range(3)
            ]
        )

        frontend_state: dict[str, torch.Tensor] = {}
        subsampling_state: dict[str, torch.Tensor] = {}
        with safe_open(args.model / "model.safetensors", framework="pt", device="cpu") as weights:
            for name in weights.keys():
                if name.startswith("feature_extractor."):
                    frontend_state[name.removeprefix("feature_extractor.")] = weights.get_tensor(name)
                elif name.startswith("subsampling_module."):
                    subsampling_state[name.removeprefix("subsampling_module.")] = weights.get_tensor(name)
        frontend.load_state_dict(frontend_state, strict=True)
        subsampling.load_state_dict(subsampling_state, strict=True)
        frontend.eval()
        subsampling.eval()

        with torch.inference_mode():
            features = frontend(waveform(args.samples))
            result = subsampling(features).float().contiguous().cpu().numpy()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    result.tofile(args.out)
    print(f"wrote {args.out}: shape={tuple(result.shape)}, values={result.size}")
    print(f"range=[{result.min():.9g}, {result.max():.9g}], mean={result.mean():.9g}")


if __name__ == "__main__":
    main()
