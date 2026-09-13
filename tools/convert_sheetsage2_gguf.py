#!/usr/bin/env python3
"""Convert SheetSage2 + its pinned MERT2 parent into one merged GGUF.

The official SheetSage2 checkpoint is an adapter: rank-64 LoRAs target all
four MERT2 attention projections. Official inference merges those adapters in
float32 before any reduced-precision cast. This converter performs the same
operation one tensor at a time so the 2.5 GB parent need not be duplicated in
RAM.

Requirements:
    python -m pip install numpy safetensors gguf
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path

import numpy as np

try:
    import gguf
    from safetensors import safe_open
except ImportError as exc:
    raise SystemExit(
        "converter dependencies are missing; install numpy safetensors gguf"
    ) from exc


MERT_REVISION = "d8ba1c745e733b3908ce6ad16ebeb17ac7600a42"
MERT_SHA256 = "e6dd2ab187d6dd62b6521cd7d8f932e237acf0c5757745a7232082e28391350d"
TOKENIZER_FINGERPRINT = "5ba3325af0344c7f"
PROJECTION_RE = re.compile(
    r"^layers\.(\d+)\.attn\.(query_proj|key_proj|value_proj|out_proj)\.weight$"
)


def tensor_name(component: str, name: str) -> str:
    """Map checkpoint names into GGML's fixed 64-byte tensor-name field."""
    # ConvNeXt paths are the only released names that exceed the limit. Keep
    # the rest descriptive and stable while shortening this repeated segment.
    name = name.replace("subsampling_module.", "sub.")
    result = f"{component}.{name}"
    if len(result.encode("utf-8")) >= 64:
        raise ValueError(f"GGML tensor name is too long: {result!r}")
    return result


def sha256(path: Path, chunk_size: int = 16 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as source:
        return json.load(source)


def require_checkpoint(directory: Path) -> Path:
    checkpoint = directory / "model.safetensors"
    if not checkpoint.is_file():
        raise ValueError(f"missing checkpoint: {checkpoint}")
    # Git-LFS pointer files fail here with a useful message instead of in the
    # safetensors parser.
    if checkpoint.stat().st_size < 1024:
        head = checkpoint.read_bytes()[:80]
        if head.startswith(b"version https://git-lfs.github.com/spec"):
            raise ValueError(f"{checkpoint} is a Git-LFS pointer; download the model weights")
    return checkpoint


def validate_configs(sheet: dict, mert: dict) -> None:
    expected = {
        "model_type": "sheetsage2",
        "weights_format": "adapter",
        "vocab_size": 31678,
        "hidden_size": 512,
        "decoder_layers": 6,
        "num_attention_heads": 8,
        "lora_rank": 64,
        "lora_alpha": 128.0,
        "tokenizer_fingerprint": TOKENIZER_FINGERPRINT,
        "base_model_revision": MERT_REVISION,
    }
    for key, value in expected.items():
        if sheet.get(key) != value:
            raise ValueError(f"unsupported SheetSage2 config {key}={sheet.get(key)!r}; expected {value!r}")
    backbone = sheet.get("backbone_config", {})
    architectural_keys = (
        "hidden_size", "intermediate_size", "num_hidden_layers",
        "num_attention_heads", "sampling_rate", "hop_length", "n_fft",
        "win_length", "num_mel_bins", "conv_depthwise_kernel_size",
        "rotary_embedding_base", "subsampling_channels", "subsampling_depths",
        "layer_norm_eps", "subsampling_layer_norm_eps", "variant",
    )
    for key in architectural_keys:
        if backbone.get(key) != mert.get(key):
            raise ValueError(f"SheetSage2/MERT2 architecture mismatch for {key}")


def output_array(array: np.ndarray, keep_f32: bool, force_f32: bool = False) -> np.ndarray:
    array = np.ascontiguousarray(array)
    if array.dtype.kind != "f":
        return array
    if keep_f32 or force_f32 or array.ndim <= 1:
        return np.ascontiguousarray(array.astype(np.float32, copy=False))
    return np.ascontiguousarray(array.astype(np.float16))


def adapter_names(layer: str, projection: str) -> tuple[str, str]:
    stem = f"adapter.layers.{layer}.attn.{projection}"
    return f"{stem}.lora_A.weight", f"{stem}.lora_B.weight"


def add_metadata(writer, sheet: dict, mert: dict, sheet_sha: str, mert_sha: str) -> None:
    writer.add_name("SheetSage2 + MERT-v2-FullSong (merged)")
    writer.add_string("yue2.component", "transcription")
    writer.add_string("yue2.transcription.architecture", "sheetsage2-mert2-fs")
    writer.add_string("yue2.transcription.tokenizer_fingerprint", TOKENIZER_FINGERPRINT)
    writer.add_string("yue2.transcription.sheetsage2_sha256", sheet_sha)
    writer.add_string("yue2.transcription.mert2_sha256", mert_sha)
    writer.add_string("yue2.transcription.mert2_revision", MERT_REVISION)
    writer.add_string("yue2.transcription.sheetsage2_config", json.dumps(sheet, separators=(",", ":")))
    writer.add_string("yue2.transcription.mert2_config", json.dumps(mert, separators=(",", ":")))
    writer.add_uint32("yue2.transcription.sample_rate", int(sheet["sampling_rate"]))
    writer.add_uint32("yue2.transcription.vocab_size", int(sheet["vocab_size"]))
    writer.add_uint32("yue2.transcription.max_output_tokens", int(sheet["max_output_seq_len"]))
    writer.add_float32("yue2.transcription.window_seconds", float(sheet["input_audio_length"]))
    writer.add_bool("yue2.transcription.lora_merged", True)


def convert(args: argparse.Namespace) -> None:
    sheet_dir = args.sheetsage.resolve()
    mert_dir = args.mert.resolve()
    sheet_checkpoint = require_checkpoint(sheet_dir)
    mert_checkpoint = require_checkpoint(mert_dir)
    sheet_config = load_json(sheet_dir / "config.json")
    mert_config = load_json(mert_dir / "config.json")
    validate_configs(sheet_config, mert_config)

    print(f"[verify] SHA-256 {mert_checkpoint}", file=sys.stderr)
    mert_digest = sha256(mert_checkpoint)
    expected_digest = sheet_config.get("base_model_sha256", MERT_SHA256)
    if not args.no_verify and mert_digest != expected_digest:
        raise ValueError(f"MERT2 SHA-256 mismatch: expected {expected_digest}, got {mert_digest}")
    sheet_digest = sha256(sheet_checkpoint)

    writer = gguf.GGUFWriter(str(args.out), "yue2-sheetsage2", use_temp_file=True)
    add_metadata(writer, sheet_config, mert_config, sheet_digest, mert_digest)

    count = 0
    merged = 0
    scale = float(sheet_config["lora_alpha"]) / int(sheet_config["lora_rank"])
    with safe_open(str(sheet_checkpoint), framework="numpy") as sheet_source, \
         safe_open(str(mert_checkpoint), framework="numpy") as mert_source:
        sheet_keys = set(sheet_source.keys())
        for index, name in enumerate(sorted(mert_source.keys()), 1):
            value = mert_source.get_tensor(name)
            match = PROJECTION_RE.match(name)
            if match:
                a_name, b_name = adapter_names(*match.groups())
                if a_name not in sheet_keys or b_name not in sheet_keys:
                    raise ValueError(f"missing SheetSage2 adapter tensors for {name}")
                # Match the official merge: float32 B @ A, scaled, accumulated
                # into the float32 parent before storage conversion.
                a = sheet_source.get_tensor(a_name).astype(np.float32)
                b = sheet_source.get_tensor(b_name).astype(np.float32)
                value = value.astype(np.float32) + (b @ a) * scale
                merged += 1
            writer.add_tensor(tensor_name("mert2", name), output_array(value, args.keep_f32))
            count += 1
            if index % 100 == 0:
                print(f"[mert2] {index}/{len(mert_source.keys())}", file=sys.stderr)

        for name in sorted(sheet_keys):
            if name.startswith("adapter."):
                continue
            value = sheet_source.get_tensor(name)
            # Layer-mixture coefficients and normalization/bias vectors remain
            # f32; matrix weights may be stored in f16 unless requested.
            writer.add_tensor(
                tensor_name("sheetsage2", name),
                output_array(value, args.keep_f32, force_f32=name == "layer_weight"),
            )
            count += 1

    expected_merged = int(sheet_config["backbone_config"]["num_hidden_layers"]) * 4
    if merged != expected_merged:
        raise ValueError(f"merged {merged} attention projections; expected {expected_merged}")

    print(f"[write] {count} tensors ({merged} LoRA-merged projections) -> {args.out}", file=sys.stderr)
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file(progress=True)
    writer.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sheetsage", type=Path, required=True, help="downloaded m-a-p/SheetSage2 directory")
    parser.add_argument("--mert", type=Path, required=True, help="downloaded m-a-p/MERT-v2-FullSong directory")
    parser.add_argument("--out", type=Path, required=True, help="output .gguf")
    parser.add_argument("--keep-f32", action="store_true", help="keep matrix weights in f32 instead of f16")
    parser.add_argument("--no-verify", action="store_true", help="skip the pinned MERT2 checksum (development only)")
    return parser.parse_args()


if __name__ == "__main__":
    try:
        convert(parse_args())
    except (OSError, ValueError) as exc:
        raise SystemExit(f"convert_sheetsage2_gguf: {exc}") from exc
