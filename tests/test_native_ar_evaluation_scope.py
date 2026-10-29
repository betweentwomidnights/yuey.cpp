"""Optional real-weight CPU check that AR validation scope cannot change training."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest
import gguf
import numpy as np

BINARY=os.environ.get('YUE2_AR_SCOPE_TEST_BINARY')
MODEL=os.environ.get('YUE2_SCOPE_TEST_MODEL')


@unittest.skipUnless(BINARY and MODEL,'Set AR binary/model paths for native CPU parity')
class ArScopeTest(unittest.TestCase):
    def check_scope(self,dora=False):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(os.environ.get('YUE2_AR_SCOPE_TEST_OUTPUT',directory))/('dora' if dora else 'lora');root.mkdir(parents=True,exist_ok=True)
            prefix=root/'prefix.i32';np.array([151851],dtype='<i4').tofile(prefix)
            paths=[]
            for i in range(4):
                path=root/f'tokens-{i}.i32';np.array([i+1,i+2],dtype='<i4').tofile(path);paths.append(path)
            base=[BINARY,'--model',MODEL,'--device','cpu','--threads','6','--identity','ar-scope-parity',
                  '--rank','1','--kl-weight','0.2','--steps','3','--eval-every','2','--save-every','2','--shuffle-samples']
            if dora:base+=['--dora']
            for i in range(4):
                kind='sample' if i<2 else 'validation' if i==2 else 'retention'
                base += [f'--{kind}-prefix',str(prefix),f'--{kind}-tokens',str(paths[i])]
            folders=[]
            for scope in ['full','heldout']:
                out=root/scope;folders.append(out)
                result=subprocess.run(base+['--eval-scope',scope,'--out',str(out)],capture_output=True,text=True,timeout=240)
                (root/(scope+'.log')).write_text(result.stdout+result.stderr)
                self.assertEqual(result.returncode,0,result.stderr[-1500:])
            def hashes(path):
                reader=gguf.GGUFReader(str(path))
                return {t.name:hashlib.sha256(t.data.tobytes()).hexdigest() for t in reader.tensors}
            for step in [2,3]:
                self.assertEqual(hashes(folders[0]/f'step-{step}.state.gguf'),hashes(folders[1]/f'step-{step}.state.gguf'))
                a=gguf.GGUFReader(str(folders[0]/f'step-{step}.state.gguf'))
                b=gguf.GGUFReader(str(folders[1]/f'step-{step}.state.gguf'))
                self.assertEqual(a.get_field('yue2.training.rng').contents(),b.get_field('yue2.training.rng').contents())
            metrics=[[json.loads(l) for l in (folder/'metrics.jsonl').read_text().splitlines()] for folder in folders]
            for history in metrics:self.assertEqual([m['step'] for m in history],[0,2,3])
            for a,b in zip(*metrics):
                self.assertEqual(a['heldout_per_sample'],b['heldout_per_sample'])
                self.assertEqual(a['retention_per_sample'],b['retention_per_sample'])
                self.assertNotIn('train_per_sample',b)

    def test_optimizer_teacher_sampling_and_heldout_scores_match(self):self.check_scope()

    def test_dora_optimizer_teacher_sampling_and_heldout_scores_match(self):self.check_scope(True)


if __name__=='__main__':unittest.main()
