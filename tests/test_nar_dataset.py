import importlib.util
from pathlib import Path
import sys
import unittest
from types import SimpleNamespace

sys.path.insert(0,str(Path(__file__).parents[1]/'tools'))
from prepare_nar_targets import source_frame_count
spec=importlib.util.spec_from_file_location('prepare_nar_dataset',Path(__file__).parents[1]/'tools/prepare_nar_dataset.py')
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)


class NarDatasetTest(unittest.TestCase):
    def test_native_decoder_trim_is_only_accepted_for_explicit_generated_edges(self):
        info=SimpleNamespace(frames=750*1920-64,samplerate=48000,duration=(750*1920-64)/48000)
        self.assertEqual(source_frame_count({},info,False),749)
        row=dict(kind='neutral replay',native_decoder_trim_samples=64)
        self.assertEqual(source_frame_count(row,info,True),750)
        for invalid,enabled in [(row,False),(dict(row,kind='real'),True),(dict(row,native_decoder_trim_samples=1920),True)]:
            with self.assertRaises(ValueError):source_frame_count(invalid,info,enabled)

    def test_targets_match_exact_core_and_preserve_example_order(self):
        a=dict(source_sha256='a',start_frame=0,frames=750,split='train')
        b=dict(source_sha256='a',start_frame=750,frames=750,split='train')
        self.assertEqual(module.match_targets([a,b],[b,a]),[(a,a),(b,b)])

    def test_wrong_offset_split_or_length_cannot_substitute(self):
        row=dict(source_sha256='a',start_frame=0,frames=750,split='validation')
        for changed in [dict(row,start_frame=1),dict(row,frames=749),dict(row,split='train')]:
            with self.assertRaises(ValueError):module.match_targets([row],[changed])

    def test_duplicate_missing_or_extra_targets_fail(self):
        row=dict(source_sha256='a',start_frame=0,frames=750,split='train')
        for targets in [[],[row,row],[row,dict(row,source_sha256='b')]]:
            with self.assertRaises(ValueError):module.match_targets([row],targets)


if __name__=='__main__':unittest.main()
