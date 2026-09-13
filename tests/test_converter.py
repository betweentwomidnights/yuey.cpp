#!/usr/bin/env python3
"""Synthetic smoke test for the SheetSage2/MERT2 GGUF converter."""

from __future__ import annotations

import json
import gc
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from gguf import GGUFReader
from safetensors.numpy import save_file

from tools.convert_sheetsage2_gguf import MERT_REVISION, TOKENIZER_FINGERPRINT, convert


class ConverterTest(unittest.TestCase):
    def test_merges_all_attention_adapters(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            sheet_dir, mert_dir = root / "sheet", root / "mert"
            sheet_dir.mkdir()
            mert_dir.mkdir()

            backbone = {
                "hidden_size": 1024,
                "intermediate_size": 4096,
                "num_hidden_layers": 24,
                "num_attention_heads": 16,
                "sampling_rate": 24000,
                "hop_length": 240,
                "n_fft": 2048,
                "win_length": 2048,
                "num_mel_bins": 128,
                "conv_depthwise_kernel_size": 31,
                "rotary_embedding_base": 10000,
                "subsampling_channels": [128, 512, 1024],
                "subsampling_depths": [3, 4, 5],
                "layer_norm_eps": 1.0e-5,
                "subsampling_layer_norm_eps": 1.0e-6,
                "variant": "fs",
            }
            sheet_config = {
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
                "base_model_sha256": "synthetic",
                "sampling_rate": 24000,
                "max_output_seq_len": 5120,
                "input_audio_length": 300.0,
                "backbone_config": backbone,
            }
            (sheet_dir / "config.json").write_text(json.dumps(sheet_config), encoding="utf-8")
            (mert_dir / "config.json").write_text(json.dumps(backbone), encoding="utf-8")

            base = {}
            adapter = {"layer_weight": np.zeros(25, dtype=np.float32)}
            for layer in range(24):
                for projection in ("query_proj", "key_proj", "value_proj", "out_proj"):
                    base[f"layers.{layer}.attn.{projection}.weight"] = np.ones((2, 2), dtype=np.float32)
                    stem = f"adapter.layers.{layer}.attn.{projection}"
                    adapter[f"{stem}.lora_A.weight"] = np.ones((1, 2), dtype=np.float32)
                    adapter[f"{stem}.lora_B.weight"] = np.ones((2, 1), dtype=np.float32)
            save_file(base, mert_dir / "model.safetensors")
            save_file(adapter, sheet_dir / "model.safetensors")

            output = root / "model.gguf"
            convert(SimpleNamespace(
                sheetsage=sheet_dir,
                mert=mert_dir,
                out=output,
                keep_f32=False,
                no_verify=True,
            ))
            reader = GGUFReader(output)
            tensors = {tensor.name: tensor for tensor in reader.tensors}
            self.assertIn("mert2.layers.0.attn.query_proj.weight", tensors)
            merged = tensors["mert2.layers.0.attn.query_proj.weight"].data
            np.testing.assert_array_equal(merged, np.full_like(merged, 3.0))
            self.assertEqual(sum(name.startswith("mert2.layers") for name in tensors), 96)
            self.assertIn("sheetsage2.layer_weight", tensors)
            self.assertTrue(all(len(name.encode("utf-8")) < 64 for name in tensors))
            # GGUFReader uses memory maps; release them before Windows removes
            # the temporary directory.
            del merged, tensors, reader
            gc.collect()


if __name__ == "__main__":
    unittest.main()
