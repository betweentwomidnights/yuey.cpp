"""Optional native encoded-cache audit checks using small synthetic GGUFs."""
import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest
import gguf
import numpy as np

BINARY=os.environ.get('YUE2_NAR_PROGRESS_TEST_BINARY')


@unittest.skipUnless(BINARY,'Set YUE2_NAR_PROGRESS_TEST_BINARY to built yue2-train')
class CacheBytesTest(unittest.TestCase):
    def test_encoded_fingerprints_validate_headers_and_detect_same_stamp_changes(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);model=root/'model.gguf';model.write_bytes(b'identity sentinel')
            def cache(name,dtype=np.float16,origin=None,frames=2):
                path=root/name;w=gguf.GGUFWriter(str(path),'yue2_nar_sample')
                w.add_string('yue2.training.model',str(origin or model));w.add_uint32('yue2.training.ar_tokens',3)
                for key in ['fixed_adapters','conditioning_adapters']:
                    w.add_key_value('yue2.training.'+key,'',gguf.GGUFValueType.STRING)
                w.add_tensor('clean',np.zeros(frames*64,dtype=np.float32))
                for key in ['keys','values']:w.add_tensor(key,np.full(3*128*8*28,.125,dtype=dtype))
                w.write_header_to_file();w.write_kv_data_to_file();w.write_tensors_to_file();w.close()
                return path
            def audit(paths,mode='bytes-v2',ok=True):
                manifest=root/'dataset.json';manifest.write_text(json.dumps(dict(format='yue2-nar-prepared-dataset-v1',
                    completed=True,rows=[dict(cache=str(p),split='train') for p in paths])))
                result=subprocess.run([BINARY,'--model',str(model),'--dataset',str(manifest),'--device','cpu','--stream-samples',
                    '--identity','cache-test','--out',str(root/'unused'),'--validate-only','--cache-fingerprint',mode],
                    capture_output=True,text=True,timeout=30)
                if not ok:
                    self.assertNotEqual(result.returncode,0,result.stdout+result.stderr);return result
                self.assertEqual(result.returncode,0,result.stdout+result.stderr)
                self.assertNotIn('model-loading',result.stdout)
                return next(json.loads(l[len('CACHE_AUDIT '):]) for l in result.stdout.splitlines() if l.startswith('CACHE_AUDIT '))
            half=cache('half.gguf');full=cache('full.gguf',np.float32)
            old_half=audit([half],'decoded-v1');old_full=audit([full],'decoded-v1')
            self.assertEqual(old_half['identity'],old_full['identity']) # Same decoded values.
            before=audit([half]);self.assertIn('nar-cache-bytes-fnv1a64-v2:',before['identity'])
            self.assertNotEqual(before['identity'],audit([full])['identity']) # Different encoded files.
            stamp=half.stat();reader=gguf.GGUFReader(str(half));offset=reader.tensors[0].data_offset;del reader
            with half.open('r+b') as stream:
                stream.seek(offset);value=stream.read(1);stream.seek(offset);stream.write(bytes([value[0]^1]))
            os.utime(half,ns=(stamp.st_atime_ns,stamp.st_mtime_ns))
            self.assertNotEqual(before['identity'],audit([half])['identity'])
            truncated=cache('truncated.gguf')
            with truncated.open('r+b') as stream:stream.truncate(truncated.stat().st_size-1)
            self.assertIn('truncated',audit([truncated],ok=False).stderr.lower())
            wrong=root/'other-model.gguf';wrong.write_bytes(b'other')
            self.assertIn('different base',audit([cache('wrong.gguf',origin=wrong)],ok=False).stderr)
            self.assertIn('differently shaped',audit([full,cache('shape.gguf',frames=3)],ok=False).stderr)
            self.assertIn('unsupported cache fingerprint',audit([full],mode='unknown',ok=False).stderr)


if __name__=='__main__':unittest.main()
