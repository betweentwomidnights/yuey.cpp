#!/usr/bin/env python3

from __future__ import annotations

import json
import gc
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from gguf import GGMLQuantizationType, GGUFReader
from safetensors.torch import save_file

from tools.convert_yue2_lora import convert


class YuE2LoraConverterTest(unittest.TestCase):
    def test_peft_names_pairs_shapes_and_storage(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "adapter_model.safetensors"
            config = root / "adapter_config.json"
            output = root / "adapter.gguf"
            a = torch.arange(16, dtype=torch.float32).reshape(2, 8)
            b = torch.arange(12, dtype=torch.float32).reshape(6, 2)
            save_file(
                {
                    "base_model.model.model.layers.0.self_attn.q_proj.lora_A.weight": a,
                    "base_model.model.model.layers.0.self_attn.q_proj.lora_B.weight": b,
                    "ignored.buffer": torch.ones(1),
                },
                source,
            )
            config.write_text(json.dumps({"r": 2, "lora_alpha": 4}), encoding="utf-8")
            converted = convert(SimpleNamespace(
                input=source,
                config=config,
                output=output,
                name="smoke",
                alpha=None,
                base_sha256="a" * 64,
                type="f16",
                overwrite=False,
            ))
            reader = GGUFReader(converted)
            tensors = {tensor.name: tensor for tensor in reader.tensors}
            self.assertEqual(set(tensors), {
                "model.layers.0.self_attn.q_proj.lora_A",
                "model.layers.0.self_attn.q_proj.lora_B",
            })
            self.assertEqual(
                tensors["model.layers.0.self_attn.q_proj.lora_A"].tensor_type,
                GGMLQuantizationType.F16,
            )
            np.testing.assert_array_equal(
                tensors["model.layers.0.self_attn.q_proj.lora_A"].data,
                a.half().numpy(),
            )
            # GGUFReader owns memory maps whose views are retained by each
            # tensor.  Release both before TemporaryDirectory removes the file
            # on Windows, where mapped files cannot be unlinked.
            del tensors
            del reader
            gc.collect()

    def test_rejects_incomplete_and_unknown_targets(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "adapter.safetensors"
            save_file({
                "model.layers.0.self_attn.q_proj.lora_A.weight": torch.ones(2, 8),
            }, source)
            args = SimpleNamespace(
                input=source, config=None, output=root / "out.gguf", name=None,
                alpha=None, base_sha256=None, type="f32", overwrite=False,
            )
            with self.assertRaisesRegex(ValueError, "incomplete"):
                convert(args)

            save_file({
                "model.layers.0.input_layernorm.lora_A.weight": torch.ones(2, 8),
                "model.layers.0.input_layernorm.lora_B.weight": torch.ones(8, 2),
            }, source)
            with self.assertRaisesRegex(ValueError, "unsupported"):
                convert(args)

    def test_invalid_fingerprint_leaves_no_partial_file(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "adapter.safetensors"
            output = root / "adapter.gguf"
            save_file({
                "lm_head.lora_A.weight": torch.ones(1, 2),
                "lm_head.lora_B.weight": torch.ones(3, 1),
            }, source)
            args = SimpleNamespace(
                input=source, config=None, output=output, name=None,
                alpha=None, base_sha256="not-a-sha", type="f32", overwrite=False,
            )
            with self.assertRaisesRegex(ValueError, "64 hexadecimal"):
                convert(args)
            self.assertFalse(output.exists())
            self.assertFalse(output.with_name(output.name + ".incomplete").exists())


if __name__ == "__main__":
    unittest.main()
