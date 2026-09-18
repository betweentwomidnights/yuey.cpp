#!/usr/bin/env python3

from __future__ import annotations

import subprocess
import sys
import unittest

from tools.model_artifacts import build_download_plan, generation_filename, profile_files


class ModelArtifactsTest(unittest.TestCase):
    def test_full_profile_is_self_contained(self):
        files = profile_files("q4_k_m", "full")
        self.assertEqual(len(files), 7)
        self.assertEqual(files[0], "yue2-3.6B-v1.0-Q4_K_M.gguf")
        self.assertIn("sheetsage2-mert2-0.7B-v1.0-F16.gguf", files)
        self.assertIn("yue2-instrumental-cot-full-v1.0-F16-LoRA.gguf", files)
        self.assertIn("yue2-realaudio-nar-v9-v1.0-F16-LoRA.gguf", files)
        self.assertIn("yue2-semantic-tokenizer-0.7B-v1.0-F16.gguf", files)

    def test_smaller_profiles_are_strict_subsets(self):
        core = profile_files(profile="core")
        transcribe = profile_files(profile="transcribe")
        full = profile_files(profile="full")
        self.assertEqual(len(core), 3)
        self.assertEqual(len(transcribe), 4)
        self.assertLess(set(core), set(transcribe))
        self.assertLess(set(transcribe), set(full))

    def test_plan_uses_one_model_family_repo(self):
        self.assertEqual(
            build_download_plan("thepatch", profile="core"),
            [("thepatch/YuE2-3B-GGUF", profile_files(profile="core"))],
        )

    def test_unpublished_encoding_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "unpublished encoding"):
            generation_filename("q5_k_m")

    def test_published_precision_tiers(self):
        self.assertEqual(
            generation_filename("bf16"), "yue2-3.6B-v1.0-BF16.gguf"
        )
        self.assertEqual(
            generation_filename("q8_0"), "yue2-3.6B-v1.0-Q8_0.gguf"
        )

    def test_python_dry_run(self):
        result = subprocess.run(
            [sys.executable, "tools/download_models.py", "--profile", "core", "--dry-run"],
            check=True, capture_output=True, text=True,
        )
        self.assertEqual(result.stdout.count("[plan]"), 3)
        self.assertIn("thepatch/YuE2-3B-GGUF", result.stdout)


if __name__ == "__main__":
    unittest.main()
