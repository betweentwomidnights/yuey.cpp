"""Optional CPU-only CLI protocol check; synthetic files require no model download."""
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
import gguf
import numpy as np

sys.path.insert(0,str(Path(__file__).parents[1]/'tools'))
from train_nar_run import NativeNarProgress

BINARY=os.environ.get('YUE2_NAR_PROGRESS_TEST_BINARY')


@unittest.skipUnless(BINARY,'Set YUE2_NAR_PROGRESS_TEST_BINARY to a built yue2-train')
class NativeNarProgressTest(unittest.TestCase):
    def test_validated_files_are_reported_before_model_loading(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);model=root/'sentinel.gguf';cache=root/'sample.gguf'
            writer=gguf.GGUFWriter(str(model),'progress_test_sentinel')
            writer.add_tensor('sentinel',np.zeros(1,dtype=np.float32))
            writer.write_header_to_file();writer.write_kv_data_to_file();writer.write_tensors_to_file();writer.close()
            writer=gguf.GGUFWriter(str(cache),'yue2_nar_sample')
            writer.add_uint32('yue2.training.ar_tokens',1)
            writer.add_string('yue2.training.model',str(model))
            # add_string intentionally drops empty strings; native cache
            # provenance requires these keys even when no adapters are present.
            writer.add_key_value('yue2.training.fixed_adapters','',gguf.GGUFValueType.STRING)
            writer.add_key_value('yue2.training.conditioning_adapters','',gguf.GGUFValueType.STRING)
            for name,count in [('clean',64),('keys',128*8*28),('values',128*8*28)]:
                writer.add_tensor(name,np.zeros(count,dtype=np.float32))
            writer.write_header_to_file();writer.write_kv_data_to_file();writer.write_tensors_to_file();writer.close()
            manifest=root/'dataset.json';manifest.write_text(json.dumps(dict(format='yue2-nar-prepared-dataset-v1',
                completed=True,rows=[dict(cache=str(cache),split='train')])))
            result=subprocess.run([BINARY,'--model',str(model),'--dataset',str(manifest),'--device','cpu',
                '--identity','progress-test','--out',str(root/'run'),'--stream-samples'],capture_output=True,text=True,timeout=30)
            # The sentinel deliberately cannot instantiate a real model. Cache
            # validation and its native event protocol must finish beforehand.
            self.assertNotEqual(result.returncode,0)
            events=[json.loads(line[len('TRAIN_EVENT '):]) for line in result.stdout.splitlines() if line.startswith('TRAIN_EVENT ')]
            self.assertEqual([(e['stage'],e['current'],e['total']) for e in events],
                             [('cache-validation',0,1),('cache-validation',1,1),('model-loading',0,1)],result.stderr)
            log=root/'native.log';log.write_text(result.stdout)
            observer=NativeNarProgress(log,400)
            self.assertEqual(observer.read()['activity'],'Loading native NAR model')


if __name__=='__main__':unittest.main()
