#!/usr/bin/env python3
"""Write exact upstream YuE2 tokenizer vectors for the native parity test."""

from __future__ import annotations

import argparse
import importlib.util
import struct
from pathlib import Path


CASES = [
    "",
    "Hello, world! It's 2026.\n",
    "I'M we'd CAN'T she'll you're I've\t  spaced\r\nnext",
    (
        "Generate a chord-annotated ABC transcription, then generate music with "
        "codec tokens from the given conditions.\n[Tags]\nindie rock, warm vocal\n"
        "[Lyrics]\n[Verse]\nCafé lights don't fade\n"
    ),
    "中文 日本語 한국어 العربية Русский देवनागरी",
    "Cafe\u0301 A\u030a ngstro\u0308m; precomposed: Café Ångström",
    "🎵👩\u200d🎤 family 👨\u200d👩\u200d👧\u200d👦 ✨",
    "X:1\r\nM:4/4\r\nK:C\r\n|: CDEF GABc :|\r\n",
    "<abc><music><extra_0><|endoftext|>",
    "space:\u00a0thin:\u2009ideographic:\u3000line:\u2028done",
]


def load_tokenizer(upstream: Path):
    source = upstream / "src" / "yue2" / "tokenization_yue2.py"
    spec = importlib.util.spec_from_file_location("yue2_reference_tokenizer", source)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not load {source}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.YuE2TextTokenizer


def blob(value: bytes) -> bytes:
    return struct.pack("<I", len(value)) + value


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--upstream", type=Path, required=True)
    parser.add_argument("--vocab", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    tokenizer = load_tokenizer(args.upstream)(args.vocab)
    output = bytearray(b"Y2TK\x01\x00\x00\x00")
    output += struct.pack("<I", len(CASES))
    for text in CASES:
        ids = tokenizer.encode(text)
        decoded = tokenizer.decode(ids)
        output += blob(text.encode("utf-8"))
        output += struct.pack("<I", len(ids))
        output += struct.pack(f"<{len(ids)}i", *ids)
        output += blob(decoded.encode("utf-8"))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes(output)
    print(f"wrote {len(CASES)} cases to {args.output}")


if __name__ == "__main__":
    main()
