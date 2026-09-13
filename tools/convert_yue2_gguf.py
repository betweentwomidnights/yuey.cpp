#!/usr/bin/env python3
"""Convert the released YuE2-3B and YuE2-VAE checkpoints to GGUF.

The output is a self-contained package for yue2.cpp: two GGUF files plus the
tokenizer and JSON sidecars needed by generation.  Conversion is streamed
through the GGUF writer so the complete 3B checkpoint is never duplicated in
memory.

Requirements:
    python -m pip install numpy torch safetensors gguf
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
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


MODEL_ARCHITECTURE = {
    "model_type": "yue2",
    "vocab_size": 184704,
    "hidden_size": 2048,
    "intermediate_size": 6144,
    "num_hidden_layers": 28,
    "num_attention_heads": 16,
    "num_key_value_heads": 8,
    "head_dim": 128,
    "max_position_embeddings": 24576,
    "latent_dim": 64,
    "max_latent_frames": 24576,
    "latent_type": "vae",
    "timestep_shift": 1.0,
    "rope_theta": 1000000,
    "rms_norm_eps": 1.0e-6,
}

VAE_ARCHITECTURE = {
    "model_type": "yue2_vae",
    "audio_channels": 2,
    "sample_rate": 48000,
    "downsampling_ratio": 1920,
    "latent_dim": 64,
    "decode_core_frames": 1024,
    "decode_halo_frames": 16,
    "release_variant": "standard",
}

MODEL_SIDECARS = (
    ("config.json", "yue2-model-config.json"),
    ("yue2_generation_config.json", "yue2-generation-config.json"),
    ("qwen.tiktoken", "yue2-qwen.tiktoken"),
)
VAE_SIDECARS = (("config.json", "yue2-vae-config.json"),)
STORAGE_TYPES = ("f32", "f16", "bf16")


def load_json(path: Path) -> dict:
    try:
        with path.open("r", encoding="utf-8") as source:
            value = json.load(source)
    except FileNotFoundError as exc:
        raise ValueError(f"missing sidecar: {path}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object in {path}")
    return value


def require_checkpoint(directory: Path) -> Path:
    checkpoint = directory / "model.safetensors"
    if not checkpoint.is_file():
        raise ValueError(f"missing checkpoint: {checkpoint}")
    if checkpoint.stat().st_size < 1024:
        head = checkpoint.read_bytes()[:80]
        if head.startswith(b"version https://git-lfs.github.com/spec"):
            raise ValueError(f"{checkpoint} is a Git-LFS pointer; download the model weights")
    return checkpoint


def validate_mapping(label: str, config: dict, expected: dict) -> None:
    for key, value in expected.items():
        if config.get(key) != value:
            raise ValueError(
                f"unsupported {label} config {key}={config.get(key)!r}; expected {value!r}"
            )


def validate_configs(model: dict, vae: dict) -> None:
    validate_mapping("YuE2-3B", model, MODEL_ARCHITECTURE)
    validate_mapping("YuE2-VAE", vae, VAE_ARCHITECTURE)
    decoder = vae.get("decoder_config", {})
    encoder = vae.get("encoder_config", {})
    expected_decoder = {
        "channels": 64,
        "latent_dim": 64,
        "out_channels": 2,
        "c_mults": [1, 2, 4, 8, 16, 32],
        "strides": [2, 2, 4, 4, 5, 6],
    }
    expected_encoder = {
        "channels": 64,
        "latent_dim": 128,
        "in_channels": 2,
        "c_mults": [1, 2, 4, 8, 16, 32],
        "strides": [2, 2, 4, 4, 5, 6],
    }
    validate_mapping("YuE2-VAE decoder", decoder, expected_decoder)
    validate_mapping("YuE2-VAE encoder", encoder, expected_encoder)


def sha256(path: Path, chunk_size: int = 16 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def tensor_name(name: str) -> str:
    if len(name.encode("utf-8")) >= 64:
        raise ValueError(f"GGML tensor name is too long: {name!r}")
    return name


def storage_array(tensor: torch.Tensor, storage_type: str) -> tuple[np.ndarray, object | None]:
    tensor = tensor.detach().cpu().contiguous()
    if not tensor.is_floating_point():
        return np.ascontiguousarray(tensor.numpy()), None
    if storage_type == "f32":
        return np.ascontiguousarray(tensor.float().numpy()), None
    if storage_type == "f16":
        return np.ascontiguousarray(tensor.half().numpy()), None
    if storage_type == "bf16":
        # NumPy has no portable bfloat16 dtype.  GGUF accepts the native raw
        # 16-bit words when the logical GGML type and original shape are given.
        raw = tensor.to(torch.bfloat16).contiguous().view(torch.uint16).numpy()
        return np.ascontiguousarray(raw), gguf.GGMLQuantizationType.BF16
    raise ValueError(f"unsupported storage type: {storage_type}")


def add_tensor(writer, name: str, tensor: torch.Tensor, storage_type: str) -> None:
    array, raw_dtype = storage_array(tensor, storage_type)
    if raw_dtype is None:
        writer.add_tensor(tensor_name(name), array)
    else:
        writer.add_tensor(
            tensor_name(name),
            array,
            raw_shape=tuple(tensor.shape),
            raw_dtype=raw_dtype,
        )


def fold_weight_norm(g: torch.Tensor, v: torch.Tensor, name: str) -> torch.Tensor:
    g = g.detach().cpu().float()
    v = v.detach().cpu().float()
    if g.ndim != v.ndim or g.shape[0] != v.shape[0]:
        raise ValueError(f"invalid weight-norm shapes for {name}: g={tuple(g.shape)}, v={tuple(v.shape)}")
    norm_dims = tuple(index for index in range(v.ndim) if index != 0)
    if any(g.shape[index] != 1 for index in norm_dims):
        raise ValueError(f"unsupported weight-norm magnitude shape for {name}: {tuple(g.shape)}")
    norm = torch.linalg.vector_norm(v, dim=norm_dims, keepdim=True)
    if bool(torch.any(norm == 0)):
        raise ValueError(f"zero weight-norm direction in {name}")
    return (g * v / norm).contiguous()


def add_common_metadata(writer, component: str, config: dict, digest: str | None) -> None:
    writer.add_string("yue2.component", component)
    writer.add_string("yue2.tensor_names", "huggingface-native")
    writer.add_string("yue2.config", json.dumps(config, separators=(",", ":"), sort_keys=True))
    if digest is not None:
        writer.add_string("yue2.checkpoint.sha256", digest)


def write_model(
    checkpoint: Path,
    output: Path,
    config: dict,
    generation_config: dict,
    storage_type: str,
    digest: str | None,
) -> int:
    writer = gguf.GGUFWriter(str(output), "yue2", use_temp_file=True)
    writer.add_name("YuE2-3B")
    add_common_metadata(writer, "generation", config, digest)
    writer.add_string(
        "yue2.generation.config",
        json.dumps(generation_config, separators=(",", ":"), sort_keys=True),
    )
    writer.add_uint32("yue2.vocab_size", config["vocab_size"])
    writer.add_uint32("yue2.context_length", config["max_position_embeddings"])
    writer.add_uint32("yue2.embedding_length", config["hidden_size"])
    writer.add_uint32("yue2.block_count", config["num_hidden_layers"])
    writer.add_uint32("yue2.attention.head_count", config["num_attention_heads"])
    writer.add_uint32("yue2.attention.head_count_kv", config["num_key_value_heads"])
    writer.add_uint32("yue2.latent_dim", config["latent_dim"])
    writer.add_float32("yue2.timestep_shift", config["timestep_shift"])

    with safe_open(str(checkpoint), framework="pt", device="cpu") as source:
        keys = sorted(source.keys())
        for index, name in enumerate(keys, 1):
            add_tensor(writer, name, source.get_tensor(name), storage_type)
            if index % 50 == 0:
                print(f"[model] {index}/{len(keys)}", file=sys.stderr)
    print(f"[write] {len(keys)} model tensors -> {output}", file=sys.stderr)
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file(progress=True)
    writer.close()
    return len(keys)


def write_vae(
    checkpoint: Path,
    output: Path,
    config: dict,
    storage_type: str,
    digest: str | None,
) -> tuple[int, int]:
    writer = gguf.GGUFWriter(str(output), "yue2_vae", use_temp_file=True)
    writer.add_name("YuE2-VAE (weight norm folded)")
    add_common_metadata(writer, "vae", config, digest)
    writer.add_uint32("yue2.vae.sample_rate", config["sample_rate"])
    writer.add_uint32("yue2.vae.audio_channels", config["audio_channels"])
    writer.add_uint32("yue2.vae.downsampling_ratio", config["downsampling_ratio"])
    writer.add_uint32("yue2.vae.latent_dim", config["latent_dim"])
    writer.add_uint32("yue2.vae.decode_core_frames", config["decode_core_frames"])
    writer.add_uint32("yue2.vae.decode_halo_frames", config["decode_halo_frames"])
    writer.add_bool("yue2.vae.weight_norm_folded", True)

    folded = 0
    written = 0
    with safe_open(str(checkpoint), framework="pt", device="cpu") as source:
        keys = sorted(source.keys())
        key_set = set(keys)
        for index, name in enumerate(keys, 1):
            if name.endswith(".weight_g"):
                v_name = name[:-len(".weight_g")] + ".weight_v"
                if v_name not in key_set:
                    raise ValueError(f"missing weight_v pair for {name}")
                continue
            if name.endswith(".weight_v"):
                base = name[:-len(".weight_v")]
                g_name = base + ".weight_g"
                if g_name not in key_set:
                    raise ValueError(f"missing weight_g pair for {name}")
                value = fold_weight_norm(source.get_tensor(g_name), source.get_tensor(name), base)
                add_tensor(writer, base + ".weight", value, storage_type)
                folded += 1
            else:
                add_tensor(writer, name, source.get_tensor(name), storage_type)
            written += 1
            if index % 50 == 0:
                print(f"[vae] {index}/{len(keys)}", file=sys.stderr)
    if folded == 0:
        raise ValueError("YuE2-VAE checkpoint has no weight-norm pairs")
    print(f"[write] {written} VAE tensors ({folded} folded pairs) -> {output}", file=sys.stderr)
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file(progress=True)
    writer.close()
    return written, folded


def copy_sidecars(model_dir: Path, vae_dir: Path, output_dir: Path) -> None:
    sidecar_dir = output_dir / "sidecars"
    sidecar_dir.mkdir(parents=True, exist_ok=True)
    for source_name, destination_name in MODEL_SIDECARS:
        source = model_dir / source_name
        if not source.is_file():
            raise ValueError(f"missing sidecar: {source}")
        shutil.copy2(source, sidecar_dir / destination_name)
    for source_name, destination_name in VAE_SIDECARS:
        source = vae_dir / source_name
        if not source.is_file():
            raise ValueError(f"missing sidecar: {source}")
        shutil.copy2(source, sidecar_dir / destination_name)


def require_sidecars(model_dir: Path, vae_dir: Path) -> None:
    for directory, entries in ((model_dir, MODEL_SIDECARS), (vae_dir, VAE_SIDECARS)):
        for source_name, _ in entries:
            source = directory / source_name
            if not source.is_file():
                raise ValueError(f"missing sidecar: {source}")


def typed_name(stem: str, storage_type: str) -> str:
    return f"{stem}-{storage_type}.gguf"


def convert(args: argparse.Namespace) -> tuple[Path, Path]:
    model_dir = args.model.resolve()
    vae_dir = args.vae.resolve()
    output_dir = args.out.resolve()
    model_checkpoint = require_checkpoint(model_dir)
    vae_checkpoint = require_checkpoint(vae_dir)
    model_config = load_json(model_dir / "config.json")
    vae_config = load_json(vae_dir / "config.json")
    generation_config = load_json(model_dir / "yue2_generation_config.json")
    validate_configs(model_config, vae_config)
    require_sidecars(model_dir, vae_dir)

    output_dir.mkdir(parents=True, exist_ok=True)
    model_output = output_dir / typed_name("yue2-3b", args.model_type)
    vae_output = output_dir / typed_name("yue2-vae", args.vae_type)
    existing = [path for path in (model_output, vae_output) if path.exists()]
    if existing and not args.overwrite:
        raise ValueError(f"output already exists (pass --overwrite): {existing[0]}")

    model_work = output_dir / (model_output.name + ".incomplete")
    vae_work = output_dir / (vae_output.name + ".incomplete")
    for path in (model_work, vae_work):
        if path.exists():
            path.unlink()

    model_digest = None
    vae_digest = None
    if not args.no_checksum:
        print(f"[checksum] {model_checkpoint}", file=sys.stderr)
        model_digest = sha256(model_checkpoint)
        print(f"[checksum] {vae_checkpoint}", file=sys.stderr)
        vae_digest = sha256(vae_checkpoint)

    try:
        write_model(
            model_checkpoint,
            model_work,
            model_config,
            generation_config,
            args.model_type,
            model_digest,
        )
        write_vae(vae_checkpoint, vae_work, vae_config, args.vae_type, vae_digest)
        copy_sidecars(model_dir, vae_dir, output_dir)
        model_work.replace(model_output)
        vae_work.replace(vae_output)
    finally:
        for path in (model_work, vae_work):
            if path.exists():
                path.unlink()
    print(f"[done] {output_dir}", file=sys.stderr)
    return model_output, vae_output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True, help="downloaded YuE2-3B directory")
    parser.add_argument("--vae", type=Path, required=True, help="downloaded YuE2-VAE directory")
    parser.add_argument("--out", type=Path, required=True, help="output package directory")
    parser.add_argument("--model-type", choices=STORAGE_TYPES, default="bf16")
    parser.add_argument("--vae-type", choices=STORAGE_TYPES, default="f16")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--no-checksum", action="store_true", help="omit source SHA-256 metadata")
    return parser.parse_args()


if __name__ == "__main__":
    try:
        convert(parse_args())
    except (OSError, ValueError, RuntimeError) as exc:
        raise SystemExit(f"convert_yue2_gguf: {exc}") from exc
