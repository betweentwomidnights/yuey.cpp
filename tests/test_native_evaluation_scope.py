"""Optional real-weight CPU parity for cheaper validation and maximum-shape preflight."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import unittest
import gguf
import numpy as np

BINARY=os.environ.get('YUE2_SCOPE_TEST_BINARY')
MODEL=os.environ.get('YUE2_SCOPE_TEST_MODEL')


@unittest.skipUnless(BINARY and MODEL,'Set YUE2_SCOPE_TEST_BINARY/MODEL for native CPU parity')
class EvaluationScopeTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp=tempfile.TemporaryDirectory();cls.root=Path(os.environ.get('YUE2_SCOPE_TEST_OUTPUT',cls.temp.name))
        cls.root.mkdir(parents=True,exist_ok=True);cls.rows=[];cls.timings={}
        for i,(split,tokens) in enumerate([('train',4),('train',6),('validation',8),('retention',5)]):
            path=cls.root/f'sample-{i}.gguf';writer=gguf.GGUFWriter(str(path),'yue2_nar_sample')
            writer.add_uint32('yue2.training.ar_tokens',tokens);writer.add_string('yue2.training.model',str(Path(MODEL).resolve()))
            for key in ['fixed_adapters','conditioning_adapters']:
                writer.add_key_value('yue2.training.'+key,'',gguf.GGUFValueType.STRING)
            writer.add_tensor('clean',np.full(128,.01*i,dtype=np.float32))
            for key in ['keys','values']:writer.add_tensor(key,np.zeros(tokens*128*8*28,dtype=np.float16))
            writer.write_header_to_file();writer.write_kv_data_to_file();writer.write_tensors_to_file();writer.close()
            cls.rows.append(dict(cache=str(path.resolve()),split=split,group=str(i)))
        cls.manifest=cls.root/'dataset.json';cls.manifest.write_text(json.dumps(dict(format='yue2-nar-prepared-dataset-v1',completed=True,rows=cls.rows)))

    @classmethod
    def tearDownClass(cls):
        (cls.root/'timings.json').write_text(json.dumps(cls.timings,indent=2))
        cls.temp.cleanup()

    def run_native(self,name,extra,fast=False):
        output=self.root/name
        inputs=['--sample',self.rows[0]['cache'],'--probe-ar-slots','8'] if fast else ['--dataset',str(self.manifest),'--group-samples']
        cmd=[BINARY,'--model',MODEL,*inputs,'--stream-samples','--shuffle-samples','--device','cpu','--threads','6',
             '--identity','scope-parity','--rank','1','--f32-projections','--seed','123','--out',str(output),*extra]
        start=time.monotonic();result=subprocess.run(cmd,capture_output=True,text=True,timeout=240)
        self.timings[name]=time.monotonic()-start
        (self.root/(name+'.log')).write_text(result.stdout+result.stderr)
        self.assertEqual(result.returncode,0,result.stderr[-3000:])
        return output

    def tensor_hashes(self,path):
        reader=gguf.GGUFReader(str(path))
        return {t.name:hashlib.sha256(t.data.tobytes()).hexdigest() for t in reader.tensors}

    def test_heldout_scope_preserves_losses_optimizer_rng_and_resumed_continuation(self):
        common=['--steps','3','--eval-every','2','--save-every','2']
        full=self.run_native('full',common+['--eval-scope','full'])
        heldout=self.run_native('heldout',common+['--eval-scope','heldout'])
        def metrics(folder):return [json.loads(l) for l in (folder/'history.jsonl').read_text().splitlines() if 'heldout_eval' in l]
        self.assertEqual([m['step'] for m in metrics(full)],[0,2,3])
        self.assertEqual([m['step'] for m in metrics(heldout)],[0,2,3])
        for a,b in zip(metrics(full),metrics(heldout)):
            self.assertEqual(a['step'],b['step'])
            for key in ['heldout_per_sample','heldout_per_sigma','retention_per_sample']:
                self.assertEqual(a[key],b[key])
            self.assertNotIn('train_eval',b)
        for step in [2,3]:
            a,b=full/f'step-{step}.state.gguf',heldout/f'step-{step}.state.gguf'
            self.assertEqual(self.tensor_hashes(a),self.tensor_hashes(b))
            self.assertEqual(gguf.GGUFReader(str(a)).get_field('yue2.training.rng').contents(),
                             gguf.GGUFReader(str(b)).get_field('yue2.training.rng').contents())
        resumed=self.run_native('resumed',['--steps','1','--resume',str(full/'step-2.state.gguf'),'--eval-scope','heldout'])
        self.assertEqual(self.tensor_hashes(full/'step-3.state.gguf'),self.tensor_hashes(resumed/'step-3.state.gguf'))
        self.assertEqual(gguf.GGUFReader(str(full/'step-3.state.gguf')).get_field('yue2.training.rng').contents(),
                         gguf.GGUFReader(str(resumed/'step-3.state.gguf')).get_field('yue2.training.rng').contents())

    def test_single_sample_preflight_matches_full_dataset_at_maximum_context(self):
        full=self.run_native('probe-full',['--probe-only'])
        fast=self.run_native('probe-fast',['--probe-only'],fast=True)
        for name in ['probe-full','probe-fast']:
            self.assertIn('8 AR slots',(self.root/(name+'.log')).read_text())
        self.assertEqual(self.tensor_hashes(full/'probe.gguf'),self.tensor_hashes(fast/'probe.gguf'))


if __name__=='__main__':unittest.main()
