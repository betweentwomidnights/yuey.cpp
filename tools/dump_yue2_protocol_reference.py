#!/usr/bin/env python3
"""Write checkpoint-native YuE2 prompt/protocol parity vectors."""

from __future__ import annotations

import argparse
import importlib.util
import struct
import sys
from pathlib import Path


CASES = [
    ("off", "ambient, cinematic", "[Instrumental]", None),
    ("melody", "acoustic pop", "[Verse]\nOne bright morning", None),
    ("full", "indie rock, warm vocal", "[Verse]\nCafé lights don't fade", "X:1\nM:4/4\nK:C\nCDEF GABc|"),
    ("melody", "电子, dreamy", "[Chorus]\n星光", "<abc> literal\r\nC2 E2 G4|"),
]


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def blob(value: bytes) -> bytes:
    return struct.pack("<I", len(value)) + value


def ids(values) -> bytes:
    values = list(values)
    return struct.pack("<I", len(values)) + struct.pack(f"<{len(values)}i", *values)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--upstream", type=Path, required=True)
    parser.add_argument("--vocab", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    package = args.upstream / "src" / "yue2"
    tokenizer_module = load_module("yue2_reference_tokenizer", package / "tokenization_yue2.py")
    protocol = load_module("yue2_reference_protocol", package / "protocol.py")
    tokenizer = tokenizer_module.YuE2TextTokenizer(args.vocab)

    output = bytearray(b"Y2PR\x01\x00\x00\x00")
    output += struct.pack("<I", len(CASES))
    for mode_index, (mode, style, lyrics, abc) in enumerate(CASES):
        request = protocol.SongRequest(style=style, lyrics=lyrics, cot=mode, abc=abc)
        abc_ids = None if abc is None else tokenizer.encode(abc)
        positive = protocol.token_prefixes(request, tokenizer)
        negative = None
        if mode == "off" or abc_ids is not None:
            negative = protocol.negative_prefix(request, tokenizer, abc_ids)
        output += struct.pack("<I", mode_index if mode_index < 3 else 1)
        output += blob(style.encode("utf-8"))
        output += blob(lyrics.encode("utf-8"))
        output += struct.pack("<I", 0 if abc is None else 1)
        if abc is not None:
            output += blob(abc.encode("utf-8"))
            output += ids(abc_ids)
        output += blob(request.text().encode("utf-8"))
        output += ids(positive)
        output += struct.pack("<I", 0 if negative is None else 1)
        if negative is not None:
            output += ids(negative)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes(output)
    print(f"wrote {len(CASES)} protocol cases to {args.output}")


if __name__ == "__main__":
    main()
