#!/usr/bin/env python3
"""Dump trustworthy SheetSage2 decoder logits for the native GGML parity test."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from safetensors import safe_open
from transformers import BartConfig
from transformers.models.bart.modeling_bart import BartDecoder


class Grammar:
    """Released SheetSage2 prompt grammar, expressed in fixed v1 token ranges."""

    def __init__(self) -> None:
        self.in_shift = True
        self.shift_run = 0
        self.payload_count = 0
        self.last_field = -1
        self.incomplete = None

    def allowed(self) -> torch.Tensor:
        mask = torch.zeros(31678, dtype=torch.bool)
        if self.payload_count > 0:
            mask[2] = True
        if (self.payload_count > 0 or self.in_shift) and self.shift_run < 4:
            mask[260:517] = True
        if self.incomplete == "rhythm":
            mask[30709:30965] = True
            return mask
        if self.incomplete == "melody":
            mask[31398:31654] = True
            mask[31654:31678] = True
            return mask
        ranges = ((0, 517, 30517), (1, 30517, 30709), (1, 30709, 30965),
                  (2, 30965, 30988), (3, 30988, 31012),
                  (4, 31037, 31398), (5, 31398, 31654))
        for field, begin, end in ranges:
            if self.last_field < field or (field == 5 and self.last_field <= field):
                mask[begin:end] = True
        return mask

    def update(self, token: int) -> bool:
        if token == 2:
            return True
        if 260 <= token < 517:
            if not self.in_shift and self.payload_count > 0:
                self.payload_count = 0
                self.last_field = -1
                self.incomplete = None
            self.in_shift = True
            self.shift_run += 1
            return False
        self.in_shift = False
        self.shift_run = 0
        self.payload_count += 1
        if 517 <= token < 30517:
            self.last_field, self.incomplete = 0, None
        elif 30517 <= token < 30709:
            self.last_field, self.incomplete = 1, "rhythm"
        elif 30709 <= token < 30965:
            self.last_field, self.incomplete = 1, None
        elif 30965 <= token < 30988:
            self.last_field, self.incomplete = 2, None
        elif 30988 <= token < 31012:
            self.last_field, self.incomplete = 3, None
        elif 31037 <= token < 31398:
            self.last_field, self.incomplete = 4, None
        elif 31398 <= token < 31654:
            self.last_field, self.incomplete = 5, "melody"
        elif 31654 <= token < 31678:
            self.last_field, self.incomplete = 5, None
        else:
            raise RuntimeError(f"unexpected generated token {token}")
        return False


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--memory", type=Path, required=True)
    parser.add_argument("--logits", type=Path, required=True)
    parser.add_argument("--ids", type=Path, required=True)
    parser.add_argument("--gguf-f16", action="store_true")
    parser.add_argument("--generated-ids", type=Path)
    parser.add_argument("--max-tokens", type=int, default=12)
    args = parser.parse_args()

    config = json.loads((args.model / "config.json").read_text(encoding="utf-8"))
    decoder_config = BartConfig(
        vocab_size=config["vocab_size"],
        d_model=config["hidden_size"],
        decoder_layers=config["decoder_layers"],
        decoder_attention_heads=config["num_attention_heads"],
        decoder_ffn_dim=config["intermediate_size"],
        max_position_embeddings=config["max_output_seq_len"],
        dropout=config["decoder_dropout"],
        attention_dropout=config["decoder_dropout"],
        activation_dropout=config["decoder_dropout"],
        activation_function="gelu",
        pad_token_id=config["pad_token_id"],
        bos_token_id=config["bos_token_id"],
        eos_token_id=config["eos_token_id"],
        is_encoder_decoder=True,
        use_cache=True,
    )
    decoder_config._attn_implementation = "sdpa"
    token_embedding = torch.nn.Embedding(config["vocab_size"], config["hidden_size"], padding_idx=0)
    decoder = BartDecoder(decoder_config).eval()
    decoder.embed_tokens = token_embedding
    with safe_open(args.model / "model.safetensors", framework="pt", device="cpu") as source:
        token_embedding.weight.data.copy_(source.get_tensor("token_embedding.weight"))
        decoder_state = {
            name.removeprefix("decoder."): source.get_tensor(name)
            for name in source.keys()
            if name.startswith("decoder.")
        }
    decoder_state["embed_tokens.weight"] = token_embedding.weight.detach()
    decoder.load_state_dict(decoder_state, strict=True)
    if args.gguf_f16:
        with torch.no_grad():
            for parameter in decoder.parameters():
                if parameter.ndim > 1:
                    parameter.copy_(parameter.half().float())

    raw_memory = np.fromfile(args.memory, dtype=np.float32)
    memory = torch.from_numpy(raw_memory.reshape(1, -1, 512).copy())
    # Canonical timestamp + full-melody prefix: SOS, timestamp, melody_full, OUT.
    ids = torch.tensor([[1, 4, 11, 3]], dtype=torch.long)
    with torch.inference_mode():
        hidden = decoder(input_ids=ids, encoder_hidden_states=memory, return_dict=True).last_hidden_state
        logits = torch.nn.functional.linear(hidden, token_embedding.weight)
    args.logits.parent.mkdir(parents=True, exist_ok=True)
    logits.float().contiguous().numpy().tofile(args.logits)
    ids.to(torch.int32).contiguous().numpy().tofile(args.ids)
    print(f"wrote logits {tuple(logits.shape)} to {args.logits}")
    print(f"range=[{logits.min().item():.9g}, {logits.max().item():.9g}]")

    if args.generated_ids is not None:
        tokens = ids[0].tolist()
        grammar = Grammar()
        while len(tokens) < args.max_tokens:
            current = torch.tensor([tokens], dtype=torch.long)
            with torch.inference_mode():
                hidden = decoder(
                    input_ids=current,
                    encoder_hidden_states=memory,
                    return_dict=True,
                ).last_hidden_state
                next_logits = torch.nn.functional.linear(hidden[:, -1], token_embedding.weight)[0]
            token = int(next_logits.masked_fill(~grammar.allowed(), float("-inf")).argmax())
            tokens.append(token)
            if grammar.update(token):
                break
        if tokens[-1] != 2:
            tokens.append(2)
        args.generated_ids.parent.mkdir(parents=True, exist_ok=True)
        np.asarray(tokens, dtype=np.int32).tofile(args.generated_ids)
        print(f"generated ids ({len(tokens)}): {tokens}")


if __name__ == "__main__":
    main()
