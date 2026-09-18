#!/usr/bin/env python3
"""Convert MERT-v2-FullSong plus a Mothersuperior semantic head to GGUF.

The head maps instance-normalised MERT layer-20 features at 25 Hz to YuE2's
32,768 semantic codec IDs.  It is kept separate from SheetSage2 because the
head was trained against the unmodified MERT parent, not SheetSage2's merged
attention adapter.
"""

from __future__ import annotations

import argparse
import json
import math
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

try:
    import gguf_meta
    from convert_sheetsage2_gguf import (
        MERT_REVISION,
        MERT_SHA256,
        MERT_SOURCE,
        load_json,
        output_array,
        require_checkpoint,
        sha256,
        tensor_name,
    )
except ImportError:
    from tools import gguf_meta
    from tools.convert_sheetsage2_gguf import (
        MERT_REVISION,
        MERT_SHA256,
        MERT_SOURCE,
        load_json,
        output_array,
        require_checkpoint,
        sha256,
        tensor_name,
    )


BASENAME = "yue2-semantic-tokenizer"
HEAD_SOURCE = (
    "yue2-mothersuperior-realaudio-tokenizer-v4",
    "Mothersuperior",
    "https://huggingface.co/Mothersuperior/yue2-mothersuperior-realaudio-tokenizer-v4",
)
INPUT = 1024
HIDDEN = 512
LAYERS = 8
HEADS = 8
FFN = 2048
WINDOW = 512
VOCAB = 32768
MERT_LAYER = 20


def expected_head_shapes() -> dict[str, tuple[int, ...]]:
    shapes: dict[str, tuple[int, ...]] = {
        "inp.weight": (HIDDEN, INPUT),
        "inp.bias": (HIDDEN,),
        "pos": (1, WINDOW, HIDDEN),
        "norm.weight": (HIDDEN,),
        "norm.bias": (HIDDEN,),
        "head.weight": (VOCAB, HIDDEN),
        "head.bias": (VOCAB,),
    }
    for layer in range(LAYERS):
        prefix = f"enc.layers.{layer}."
        shapes.update({
            prefix + "self_attn.in_proj_weight": (3 * HIDDEN, HIDDEN),
            prefix + "self_attn.in_proj_bias": (3 * HIDDEN,),
            prefix + "self_attn.out_proj.weight": (HIDDEN, HIDDEN),
            prefix + "self_attn.out_proj.bias": (HIDDEN,),
            prefix + "linear1.weight": (FFN, HIDDEN),
            prefix + "linear1.bias": (FFN,),
            prefix + "linear2.weight": (HIDDEN, FFN),
            prefix + "linear2.bias": (HIDDEN,),
            prefix + "norm1.weight": (HIDDEN,),
            prefix + "norm1.bias": (HIDDEN,),
            prefix + "norm2.weight": (HIDDEN,),
            prefix + "norm2.bias": (HIDDEN,),
        })
    return shapes


def validate_head(source) -> None:
    expected = expected_head_shapes()
    actual = set(source.keys())
    missing = sorted(set(expected) - actual)
    extra = sorted(actual - set(expected))
    if missing or extra:
        raise ValueError(
            f"unsupported semantic head tensor set: missing={missing[:4]}, extra={extra[:4]}"
        )
    for name, shape in expected.items():
        actual_shape = tuple(source.get_slice(name).get_shape())
        if actual_shape != shape:
            raise ValueError(
                f"semantic head shape mismatch for {name}: {actual_shape} != {shape}"
            )


def head_tensor_name(name: str) -> str:
    result = "semantic_head." + name
    if len(result.encode("utf-8")) >= 64:
        raise ValueError(f"GGML tensor name is too long: {result}")
    return result


def resolve_output(out: Path, storage: str, parameters: int) -> Path:
    if out.suffix == ".gguf":
        return out
    out.mkdir(parents=True, exist_ok=True)
    return out / gguf_meta.gguf_filename(BASENAME, storage, parameters)


def validate_mert_config(config: dict) -> None:
    if int(config.get("sampling_rate", 0)) != 24000 or \
       int(config.get("hidden_size", 0)) != INPUT or \
       int(config.get("num_hidden_layers", 0)) <= MERT_LAYER:
        raise ValueError("unsupported MERT-v2-FullSong configuration")


def convert(args: argparse.Namespace) -> Path:
    mert_dir = args.mert.resolve()
    head_path = args.head.resolve()
    mert_checkpoint = require_checkpoint(mert_dir)
    if not head_path.is_file():
        raise ValueError(f"missing semantic tokenizer head: {head_path}")
    mert_config = load_json(mert_dir / "config.json")
    validate_mert_config(mert_config)

    print(f"[verify] SHA-256 {mert_checkpoint}", file=sys.stderr)
    mert_digest = sha256(mert_checkpoint)
    if not args.no_verify and mert_digest != MERT_SHA256:
        raise ValueError(
            f"MERT2 SHA-256 mismatch: expected {MERT_SHA256}, got {mert_digest}"
        )
    head_digest = sha256(head_path)

    with safe_open(str(head_path), framework="pt", device="cpu") as head:
        validate_head(head)
        head_parameters = sum(math.prod(head.get_slice(name).get_shape()) for name in head.keys())
    with safe_open(str(mert_checkpoint), framework="numpy") as mert:
        mert_parameters = sum(math.prod(mert.get_slice(name).get_shape()) for name in mert.keys())

    storage = "f32" if args.keep_f32 else "f16"
    parameters = mert_parameters + head_parameters
    output = resolve_output(args.out.resolve(), storage, parameters)
    writer = gguf.GGUFWriter(str(output), "yue2-semantic-tokenizer", use_temp_file=True)
    gguf_meta.add_general(
        writer, BASENAME, "MERT-v2-FullSong + YuE2 semantic tokenizer head",
        n_params=parameters,
    )
    gguf_meta.add_sources(writer, [(*MERT_SOURCE, MERT_REVISION), (*HEAD_SOURCE, None)])
    gguf_meta.add_file_type(writer, storage)
    writer.add_string("yue2.component", "semantic-tokenizer")
    writer.add_string(
        "yue2.semantic_tokenizer.architecture", "mert2-layer20-transformer-v1"
    )
    writer.add_string("yue2.semantic_tokenizer.mert2_revision", MERT_REVISION)
    writer.add_string("yue2.semantic_tokenizer.mert2_sha256", mert_digest)
    writer.add_string("yue2.semantic_tokenizer.head_sha256", head_digest)
    writer.add_string(
        "yue2.semantic_tokenizer.mert2_config",
        json.dumps(mert_config, separators=(",", ":")),
    )
    writer.add_uint32("yue2.semantic_tokenizer.sample_rate", 24000)
    writer.add_uint32("yue2.semantic_tokenizer.frame_rate", 25)
    writer.add_uint32("yue2.semantic_tokenizer.mert_layer", MERT_LAYER)
    writer.add_uint32("yue2.semantic_tokenizer.input_size", INPUT)
    writer.add_uint32("yue2.semantic_tokenizer.hidden_size", HIDDEN)
    writer.add_uint32("yue2.semantic_tokenizer.layer_count", LAYERS)
    writer.add_uint32("yue2.semantic_tokenizer.head_count", HEADS)
    writer.add_uint32("yue2.semantic_tokenizer.window_frames", WINDOW)
    writer.add_uint32("yue2.semantic_tokenizer.vocab_size", VOCAB)
    writer.add_bool("yue2.semantic_tokenizer.instance_normalized", True)

    with safe_open(str(mert_checkpoint), framework="numpy") as mert:
        for index, name in enumerate(sorted(mert.keys()), 1):
            writer.add_tensor(
                tensor_name("mert2", name),
                output_array(mert.get_tensor(name), args.keep_f32),
            )
            if index % 100 == 0:
                print(f"[mert2] {index}/{len(mert.keys())}", file=sys.stderr)

    with safe_open(str(head_path), framework="pt", device="cpu") as head:
        for name in sorted(head.keys()):
            tensor = head.get_tensor(name).detach().cpu().contiguous()
            if tensor.ndim <= 1 or args.keep_f32:
                array = tensor.float().numpy()
            else:
                array = tensor.half().numpy()
            writer.add_tensor(head_tensor_name(name), np.ascontiguousarray(array))

    print(f"[write] MERT2 + {len(expected_head_shapes())} head tensors -> {output}", file=sys.stderr)
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file(progress=True)
    writer.close()
    return output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mert", type=Path, required=True,
                        help="downloaded m-a-p/MERT-v2-FullSong directory")
    parser.add_argument("--head", type=Path, required=True,
                        help="Mothersuperior tokenizer_head_*.safetensors")
    parser.add_argument("--out", type=Path, required=True,
                        help="output directory or explicit .gguf path")
    parser.add_argument("--keep-f32", action="store_true")
    parser.add_argument("--no-verify", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    try:
        convert(parse_args())
    except (OSError, ValueError, RuntimeError) as exc:
        raise SystemExit(f"convert_semantic_tokenizer_gguf: {exc}") from exc
