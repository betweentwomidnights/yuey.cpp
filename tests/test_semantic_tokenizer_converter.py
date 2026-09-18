#!/usr/bin/env python3

from __future__ import annotations

import unittest

from tools.convert_semantic_tokenizer_gguf import (
    expected_head_shapes,
    head_tensor_name,
    validate_head,
    validate_mert_config,
)


class _Slice:
    def __init__(self, shape):
        self._shape = shape

    def get_shape(self):
        return self._shape


class _Source:
    def __init__(self, shapes):
        self.shapes = shapes

    def keys(self):
        return self.shapes.keys()

    def get_slice(self, name):
        return _Slice(self.shapes[name])


class SemanticTokenizerConverterTest(unittest.TestCase):
    def test_released_head_contract(self):
        shapes = expected_head_shapes()
        self.assertEqual(len(shapes), 103)
        self.assertEqual(shapes["inp.weight"], (512, 1024))
        self.assertEqual(shapes["pos"], (1, 512, 512))
        self.assertEqual(
            shapes["enc.layers.7.self_attn.in_proj_weight"], (1536, 512)
        )
        self.assertEqual(shapes["head.weight"], (32768, 512))
        validate_head(_Source(shapes))

    def test_rejects_shape_or_tensor_set_drift(self):
        shapes = expected_head_shapes()
        shapes["inp.weight"] = (511, 1024)
        with self.assertRaisesRegex(ValueError, "shape mismatch"):
            validate_head(_Source(shapes))
        shapes = expected_head_shapes()
        shapes["unexpected"] = (1,)
        with self.assertRaisesRegex(ValueError, "unsupported semantic head tensor set"):
            validate_head(_Source(shapes))

    def test_ggml_names_fit_legacy_name_limit(self):
        for name in expected_head_shapes():
            converted = head_tensor_name(name)
            self.assertTrue(converted.startswith("semantic_head."))
            self.assertLess(len(converted.encode("utf-8")), 64)

    def test_mert_config_must_include_layer_twenty(self):
        valid = {
            "sampling_rate": 24000,
            "hidden_size": 1024,
            "num_hidden_layers": 21,
        }
        validate_mert_config(valid)
        invalid = dict(valid, num_hidden_layers=20)
        with self.assertRaisesRegex(ValueError, "unsupported MERT"):
            validate_mert_config(invalid)


if __name__ == "__main__":
    unittest.main()
