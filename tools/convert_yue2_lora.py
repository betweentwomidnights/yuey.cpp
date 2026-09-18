#!/usr/bin/env python3
"""Convert a PEFT-style YuE2 LoRA safetensors checkpoint to runtime GGUF."""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np

try:
    import gguf
    import torch
    from safetensors import safe_open
except ImportError as exc:
    raise SystemExit(
        "converter dependencies are missing; install numpy torch safetensors gguf"
    ) from exc


TARGET = re.compile(
    r"(?:model\.layers\.\d+\."
    r"(?:(?:nar_)?self_attn\.(?:q|k|v|o)_proj|(?:nar_)?mlp\.(?:gate|up|down)_proj)"
    r"|lm_head|vae2llm|llm2vae|time_embedder\.mlp\.(?:0|2))"
)


def load_config(path: Path | None) -> dict:
    if path is None:
        return {}
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("adapter config must be a JSON object")
    return value


def strip_wrapper_prefix(name: str) -> str:
    for prefix in ("module.", "base_model.model.", "base_model."):
        if name.startswith(prefix):
            name = name[len(prefix):]
    return name


def canonical_key(name: str) -> tuple[str, str] | None:
    name = strip_wrapper_prefix(name)
    for suffix, factor in (
        (".lora_A.weight", "lora_A"),
        (".lora_B.weight", "lora_B"),
        (".lora_A", "lora_A"),
        (".lora_B", "lora_B"),
    ):
        if name.endswith(suffix):
            stem = name[: -len(suffix)]
            # Mothersuperior checkpoints use layers.N... while the released
            # YuE2 checkpoint and our GGUF runtime use model.layers.N....
            if stem.startswith("layers."):
                stem = "model." + stem
            if not TARGET.fullmatch(stem):
                raise ValueError(f"unsupported YuE2 LoRA target: {stem}")
            return stem, factor
    return None


def canonical_replacement(name: str) -> str | None:
    name = strip_wrapper_prefix(name)
    if name in {
        "vae2llm.weight", "vae2llm.bias",
        "llm2vae.weight", "llm2vae.bias",
    }:
        return name
    return None


def convert(args: argparse.Namespace) -> Path:
    source = args.input.resolve()
    output = args.output.resolve()
    if not source.is_file():
        raise ValueError(f"missing adapter checkpoint: {source}")
    if output.exists() and not args.overwrite:
        raise ValueError(f"output already exists (pass --overwrite): {output}")
    config = load_config(args.config.resolve() if args.config else None)

    pairs: dict[str, dict[str, torch.Tensor]] = {}
    replacements: dict[str, torch.Tensor] = {}
    with safe_open(source, framework="pt", device="cpu") as checkpoint:
        for source_name in checkpoint.keys():
            canonical = canonical_key(source_name)
            if canonical is not None:
                stem, factor = canonical
                pairs.setdefault(stem, {})[factor] = checkpoint.get_tensor(source_name)
                continue
            replacement = canonical_replacement(source_name)
            if replacement is not None:
                if replacement in replacements:
                    raise ValueError(f"duplicate replacement tensor: {replacement}")
                replacements[replacement] = checkpoint.get_tensor(source_name)
    if not pairs:
        raise ValueError("checkpoint contains no supported YuE2 LoRA factors")

    ranks: set[int] = set()
    for stem, factors in pairs.items():
        if set(factors) != {"lora_A", "lora_B"}:
            raise ValueError(f"incomplete LoRA factor pair for {stem}")
        a = factors["lora_A"]
        b = factors["lora_B"]
        if a.ndim != 2 or b.ndim != 2 or a.shape[0] != b.shape[1]:
            raise ValueError(
                f"invalid LoRA shapes for {stem}: A={tuple(a.shape)}, B={tuple(b.shape)}"
            )
        ranks.add(int(a.shape[0]))
    if len(ranks) != 1:
        raise ValueError(f"mixed LoRA ranks are not supported: {sorted(ranks)}")
    rank = ranks.pop()
    for name, tensor in replacements.items():
        expected_dims = 1 if name.endswith(".bias") else 2
        if tensor.ndim != expected_dims:
            raise ValueError(
                f"invalid replacement shape for {name}: {tuple(tensor.shape)}"
            )
    declared_rank = int(config.get("r", rank))
    if declared_rank != rank:
        raise ValueError(f"adapter config rank {declared_rank} != tensor rank {rank}")
    alpha = float(args.alpha if args.alpha is not None else config.get("lora_alpha", rank))
    if not np.isfinite(alpha) or alpha <= 0:
        raise ValueError("LoRA alpha must be finite and positive")
    digest = None
    if args.base_sha256:
        digest = args.base_sha256.lower()
        if not re.fullmatch(r"[0-9a-f]{64}", digest):
            raise ValueError("--base-sha256 must contain 64 hexadecimal characters")

    output.parent.mkdir(parents=True, exist_ok=True)
    work = output.with_name(output.name + ".incomplete")
    if work.exists():
        work.unlink()
    writer = None
    try:
        writer = gguf.GGUFWriter(str(work), "yue2_lora", use_temp_file=True)
        writer.add_name(args.name or source.stem)
        writer.add_string("yue2.component", "generation-adapter")
        writer.add_string("yue2.adapter.type", "lora")
        writer.add_uint32("yue2.adapter.rank", rank)
        writer.add_float32("yue2.adapter.alpha", alpha)
        writer.add_uint32("yue2.adapter.target_count", len(pairs) + len(replacements))
        writer.add_uint32("yue2.adapter.replacement_count", len(replacements))
        if digest:
            writer.add_string("yue2.adapter.base_sha256", digest)
        for stem in sorted(pairs):
            for factor in ("lora_A", "lora_B"):
                tensor = pairs[stem][factor].detach().cpu().contiguous()
                tensor = tensor.float() if args.type == "f32" else tensor.half()
                name = f"{stem}.{factor}"
                if len(name.encode("utf-8")) >= 64:
                    raise ValueError(f"GGML tensor name is too long: {name}")
                writer.add_tensor(name, np.ascontiguousarray(tensor.numpy()))
        for base_name in sorted(replacements):
            tensor = replacements[base_name].detach().cpu().contiguous()
            tensor = tensor.float() if args.type == "f32" else tensor.half()
            name = f"{base_name}.replacement"
            if len(name.encode("utf-8")) >= 64:
                raise ValueError(f"GGML tensor name is too long: {name}")
            writer.add_tensor(name, np.ascontiguousarray(tensor.numpy()))
        writer.write_header_to_file()
        writer.write_kv_data_to_file()
        writer.write_tensors_to_file(progress=True)
        writer.close()
        writer = None
        work.replace(output)
    finally:
        if writer is not None:
            try:
                writer.close()
            except Exception:
                pass
        if work.exists():
            work.unlink()
    print(
        f"[done] {len(pairs)} LoRA targets + {len(replacements)} replacements, "
        f"rank {rank}, alpha {alpha:g} -> {output}",
        file=sys.stderr,
    )
    return output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="adapter_model.safetensors")
    parser.add_argument("--config", type=Path, help="optional PEFT adapter_config.json")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--name")
    parser.add_argument("--alpha", type=float, help="override lora_alpha")
    parser.add_argument("--base-sha256", help="fingerprint of the intended YuE2 GGUF source checkpoint")
    parser.add_argument("--type", choices=("f32", "f16"), default="f32")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    try:
        convert(parse_args())
    except (OSError, ValueError, RuntimeError) as exc:
        raise SystemExit(f"convert_yue2_lora: {exc}") from exc
