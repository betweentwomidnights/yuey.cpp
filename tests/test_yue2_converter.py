#!/usr/bin/env python3
"""Synthetic smoke tests for the YuE2 generation/VAE GGUF converter."""

from __future__ import annotations

import gc
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from gguf import GGMLQuantizationType, GGUFReader
from safetensors.torch import save_file

from tools import gguf_meta
from tools.convert_yue2_gguf import (
    MODEL_ARCHITECTURE,
    VAE_ARCHITECTURE,
    convert,
)


def string_field(reader: GGUFReader, key: str) -> str:
    field = reader.fields[key]
    return bytes(field.parts[field.data[0]]).decode("utf-8")


def model_config() -> dict:
    return dict(MODEL_ARCHITECTURE)


def vae_config() -> dict:
    return {
        **VAE_ARCHITECTURE,
        "decoder_config": {
            "channels": 64,
            "latent_dim": 64,
            "out_channels": 2,
            "c_mults": [1, 2, 4, 8, 16, 32],
            "strides": [2, 2, 4, 4, 5, 6],
        },
        "encoder_config": {
            "channels": 64,
            "latent_dim": 128,
            "in_channels": 2,
            "c_mults": [1, 2, 4, 8, 16, 32],
            "strides": [2, 2, 4, 4, 5, 6],
        },
    }


class YuE2ConverterTest(unittest.TestCase):
    def test_preserves_bf16_and_folds_vae_weight_norm(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            model_dir = root / "YuE2-3B"
            vae_dir = root / "YuE2-VAE"
            output_dir = root / "output"
            model_dir.mkdir()
            vae_dir.mkdir()

            (model_dir / "config.json").write_text(json.dumps(model_config()), encoding="utf-8")
            (model_dir / "yue2_generation_config.json").write_text(
                json.dumps({"do_sample": True, "temperature": 1.0}), encoding="utf-8"
            )
            (model_dir / "qwen.tiktoken").write_text("YQ== 0\n", encoding="utf-8")
            (vae_dir / "config.json").write_text(json.dumps(vae_config()), encoding="utf-8")

            model_weights = {
                "model.embed_tokens.weight": torch.tensor(
                    [[1.0, -2.0], [3.0, 4.0]], dtype=torch.bfloat16
                ),
                "model.norm.weight": torch.tensor([0.5, 1.5], dtype=torch.bfloat16),
            }
            save_file(model_weights, model_dir / "model.safetensors")

            direction = torch.tensor(
                [[[3.0, 4.0]], [[0.0, 2.0]]], dtype=torch.float32
            )
            magnitude = torch.tensor([[[10.0]], [[3.0]]], dtype=torch.float32)
            save_file(
                {
                    "decoder.layers.0.bias": torch.tensor([1.0, -1.0]),
                    "decoder.layers.0.weight_g": magnitude,
                    "decoder.layers.0.weight_v": direction,
                },
                vae_dir / "model.safetensors",
            )

            model_output, vae_output = convert(SimpleNamespace(
                model=model_dir,
                vae=vae_dir,
                out=output_dir,
                model_type="bf16",
                vae_type="f16",
                model_revision="0123abcd",
                vae_revision=None,
                overwrite=False,
                no_checksum=True,
            ))
            # Six parameters: the 2x2 embedding and the 2-element norm.
            self.assertEqual(model_output.name, gguf_meta.gguf_filename("yue2", "bf16", 6))
            self.assertTrue(model_output.name.endswith("-v1.0-BF16.gguf"))
            self.assertEqual(vae_output.name, "yue2-vae-v1.0-F16.gguf")

            with self.assertRaisesRegex(ValueError, "output already exists"):
                convert(SimpleNamespace(
                    model=model_dir,
                    vae=vae_dir,
                    out=output_dir,
                    model_type="bf16",
                    vae_type="f16",
                    overwrite=False,
                    no_checksum=True,
                ))

            model_reader = GGUFReader(model_output)
            vae_reader = GGUFReader(vae_output)
            model_tensors = {tensor.name: tensor for tensor in model_reader.tensors}
            vae_tensors = {tensor.name: tensor for tensor in vae_reader.tensors}
            try:
                self.assertEqual(string_field(model_reader, "general.basename"), "yue2")
                self.assertEqual(string_field(model_reader, "general.license"), "cc-by-nc-4.0")
                self.assertEqual(string_field(model_reader, "general.version"), "v1.0")
                self.assertEqual(
                    string_field(model_reader, "general.base_model.0.repo_url"),
                    "https://huggingface.co/m-a-p/YuE2-3B",
                )
                self.assertEqual(string_field(model_reader, "general.base_model.0.version"), "0123abcd")
                self.assertEqual(int(model_reader.fields["general.file_type"].parts[-1][0]), 32)
                self.assertEqual(string_field(vae_reader, "general.basename"), "yue2-vae")
                self.assertNotIn("general.size_label", vae_reader.fields)
                embedding = model_tensors["model.embed_tokens.weight"]
                self.assertEqual(embedding.tensor_type, GGMLQuantizationType.BF16)
                actual_words = np.frombuffer(embedding.data.tobytes(), dtype=np.uint16)
                expected_words = model_weights["model.embed_tokens.weight"].view(torch.uint16).numpy().ravel()
                np.testing.assert_array_equal(actual_words, expected_words)

                self.assertIn("decoder.layers.0.weight", vae_tensors)
                self.assertNotIn("decoder.layers.0.weight_g", vae_tensors)
                self.assertNotIn("decoder.layers.0.weight_v", vae_tensors)
                expected = np.array([[[6.0, 8.0]], [[0.0, 3.0]]], dtype=np.float16)
                np.testing.assert_array_equal(vae_tensors["decoder.layers.0.weight"].data, expected)
                self.assertTrue((output_dir / "sidecars" / "yue2-qwen.tiktoken").is_file())
                self.assertTrue(all(
                    len(name.encode("utf-8")) < 64 for name in model_tensors | vae_tensors
                ))
            finally:
                del embedding, model_tensors, vae_tensors, model_reader, vae_reader
                gc.collect()


if __name__ == "__main__":
    unittest.main()
