import importlib.util
from pathlib import Path
import sys
import unittest
import tempfile
import gguf
import numpy as np
from types import SimpleNamespace

sys.path.insert(0,str(Path(__file__).parents[1]/'tools'))
from prepare_nar_targets import source_frame_count
spec=importlib.util.spec_from_file_location('prepare_nar_dataset',Path(__file__).parents[1]/'tools/prepare_nar_dataset.py')
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)


class NarDatasetTest(unittest.TestCase):
    def test_native_cache_payload_provenance_and_finiteness(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);base=root/'base';common=root/'common';adapter=root/'ar'
            prefix=np.array([1,2],dtype='<i4');codes=np.array([3,4],dtype='<i4');clean=np.zeros((2,64),dtype='<f4')
            prefix.tofile(root/'prefix');codes.tofile(root/'codes');clean.tofile(root/'means')
            row=dict(prefix=str(root/'prefix'),codes=str(root/'codes'),frames=2);target=dict(means=str(root/'means'))
            def cache(name,bad_prefix=False,nonfinite=False):
                path=root/(name+'.gguf');writer=gguf.GGUFWriter(str(path),'yue2_nar_sample')
                writer.add_string('yue2.training.cache.identity','key');writer.add_string('yue2.training.model',str(base))
                writer.add_key_value('yue2.training.fixed_adapters','',gguf.GGUFValueType.STRING)
                writer.add_string('yue2.training.conditioning_adapters',''.join(str(p.resolve())+'=1.000000\n' for p in [common,adapter]))
                writer.add_uint32('yue2.training.ar_tokens',5)
                keys=np.zeros((28,8,5,128),dtype=np.float16)
                if nonfinite:keys.flat[0]=np.nan
                for key,value in [('prefix',prefix[::-1].copy() if bad_prefix else prefix),('codes',codes),('clean',clean),('keys',keys),('values',keys.copy())]:writer.add_tensor(key,value)
                writer.write_header_to_file();writer.write_kv_data_to_file();writer.write_tensors_to_file();writer.close()
                return path
            module.verify_cache(cache('valid'),'key',row,target,base,common,adapter,'f16')
            with self.assertRaisesRegex(ValueError,'input payload differs'):
                module.verify_cache(cache('wrong-prefix',bad_prefix=True),'key',row,target,base,common,adapter,'f16')
            with self.assertRaisesRegex(ValueError,'Nonfinite'):
                module.verify_cache(cache('nan',nonfinite=True),'key',row,target,base,common,adapter,'f16')

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
