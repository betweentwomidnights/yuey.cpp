"""Checks that feasibility metrics and split/provenance handling are meaningful."""
import argparse
import contextlib
import importlib.util
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np
import soundfile as sf

sys.path.insert(0, str(Path(__file__).parents[1]/"tools"))
spec = importlib.util.spec_from_file_location("audit", Path(__file__).parents[1]/"tools/yue2_training_audit.py")
audit = importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit)


class AuditTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def audio(self, path):
        t = np.arange(48000) / 24000
        envelope = .1 + .2 * (np.sin(2*np.pi*3*t)>0)
        sf.write(path, np.column_stack([envelope*np.sin(2*np.pi*440*t),
                                       envelope*np.sin(2*np.pi*660*t)]), 24000, subtype="FLOAT")

    def test_identical_and_wrong_tone_controls(self):
        a = self.root/"a.wav"; self.audio(a)
        same = audit.compare(a, a)
        self.assertAlmostEqual(same["chroma_correlation"], 1)
        self.assertAlmostEqual(same["onset_correlation"], 1)
        self.assertEqual(same["log_spectral_l1"], 0)
        t = np.arange(48000)/24000
        b = self.root/"b.wav"; sf.write(b, .1*np.sin(2*np.pi*370*t), 24000, subtype="FLOAT")
        wrong = audit.compare(a, b)
        self.assertLess(wrong["chroma_correlation"], .5)
        self.assertGreater(wrong["log_spectral_l1"], 0)

    def test_tokens_reject_partial_and_out_of_range(self):
        p = self.root/"codes.i32"
        p.write_bytes(b"\x00")
        with self.assertRaises(ValueError): audit.token_stats(p)
        np.array([32768], dtype="<i4").tofile(p)
        with self.assertRaises(ValueError): audit.token_stats(p)
        np.array([3,3,4], dtype="<i4").tofile(p)
        self.assertEqual(audit.token_stats(p)["adjacent_repeat_fraction"], .5)

    def test_duplicate_audio_stays_in_one_split_and_source_unchanged(self):
        a, b = self.root/"a", self.root/"b"; a.mkdir(); b.mkdir()
        self.audio(a/"track.wav")
        (b/"renamed.wav").write_bytes((a/"track.wav").read_bytes())
        before = audit.digest(a/"track.wav")
        args = argparse.Namespace(dataset=[f"a={a}", f"b={b}"], out=self.root/"probes",
                                  seconds=1, offset=.25, count=1)
        with contextlib.redirect_stdout(io.StringIO()):
            audit.prepare(args)
        tracks = json.loads((args.out/"inventory.json").read_text())
        self.assertEqual(tracks[0]["split"], tracks[1]["split"])
        self.assertEqual(audit.digest(a/"track.wav"), before)
        probes = json.loads((args.out/"probes.json").read_text())
        for p in probes:
            self.assertEqual(audit.digest(Path(p["probe"])/"source.wav"), p["excerpt_sha256"])
        with self.assertRaises(ValueError): audit.prepare(args)


if __name__ == "__main__": unittest.main()
