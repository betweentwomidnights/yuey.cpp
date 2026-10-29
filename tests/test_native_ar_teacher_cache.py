"""Optional real-weight check that streamed AR teacher caches cannot change training.

Set YUE2_AR_TEACHER_TEST_BINARY and YUE2_AR_TEACHER_TEST_MODEL (BF16 base).
YUE2_AR_TEACHER_TEST_DEVICE defaults to cpu; cuda checks the same parity.
"""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest
import gguf
import numpy as np

BINARY=os.environ.get('YUE2_AR_TEACHER_TEST_BINARY')
MODEL=os.environ.get('YUE2_AR_TEACHER_TEST_MODEL')
DEVICE=os.environ.get('YUE2_AR_TEACHER_TEST_DEVICE','cpu')


def tensors(path):
    reader=gguf.GGUFReader(str(path))
    return {t.name:hashlib.sha256(t.data.tobytes()).hexdigest() for t in reader.tensors}


def field(path,name):
    return gguf.GGUFReader(str(path)).get_field(name).contents()


@unittest.skipUnless(BINARY and MODEL,'Set AR binary/model paths for native teacher-cache parity')
class ArTeacherCacheTest(unittest.TestCase):
    def run_native(self,root,name,args,ok=True):
        result=subprocess.run([BINARY,*map(str,args)],capture_output=True,text=True,timeout=600)
        (root/(name+'.log')).write_text(result.stdout+result.stderr)
        if ok:self.assertEqual(result.returncode,0,result.stderr[-1500:])
        return result

    def check(self,dora=False):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(os.environ.get('YUE2_AR_TEACHER_TEST_OUTPUT',directory))/('dora' if dora else 'lora')
            root.mkdir(parents=True,exist_ok=True)
            prefix=root/'prefix.i32';np.array([151851],dtype='<i4').tofile(prefix)
            base=['--model',MODEL,'--device',DEVICE,'--threads','6','--identity','ar-teacher-parity',
                  '--rank','1','--kl-weight','0.2','--eval-every','2','--save-every','2','--shuffle-samples']
            if dora:base+=['--dora']
            for i in range(5):
                path=root/f'tokens-{i}.i32';np.array([i+1,i+2],dtype='<i4').tofile(path)
                kind='sample' if i<3 else 'validation' if i==3 else 'retention'
                base+=[f'--{kind}-prefix',prefix,f'--{kind}-tokens',path]
            cache=root/'teacher'
            teacher=['--teacher-cache',cache,'--teacher-identity','fixture-identity','--teacher-base-sha256','fixture-base']
            self.run_native(root,'memory',base+['--steps',3,'--out',root/'memory'])
            prepared=self.run_native(root,'prepare',base+teacher+['--prepare-teacher-only'])
            self.assertIn('teacher_cache_ready=3',prepared.stdout)
            files=sorted(cache.iterdir())
            self.assertEqual([p.name for p in files],[f'teacher-{i:05d}.gguf' for i in range(3)])
            stamps=[(p.stat().st_mtime_ns,hashlib.sha256(p.read_bytes()).hexdigest()) for p in files]
            # Preparation is resumable and never rewrites completed files.
            self.run_native(root,'prepare-again',base+teacher+['--prepare-teacher-only'])
            self.assertEqual(stamps,[(p.stat().st_mtime_ns,hashlib.sha256(p.read_bytes()).hexdigest()) for p in files])
            streamed=self.run_native(root,'cache',base+teacher+['--steps',3,'--out',root/'cache'])
            self.assertNotIn('prepare_teacher',streamed.stdout)
            self.assertIn('teacher_cache=3/3',streamed.stdout)
            for step in [2,3]:
                a=root/'memory'/f'step-{step}.state.gguf';b=root/'cache'/f'step-{step}.state.gguf'
                self.assertEqual(tensors(a),tensors(b))
                self.assertEqual(field(a,'yue2.training.rng'),field(b,'yue2.training.rng'))
                self.assertEqual(tensors(root/'memory'/f'step-{step}.gguf'),tensors(root/'cache'/f'step-{step}.gguf'))
            self.assertEqual((root/'memory/metrics.jsonl').read_text(),(root/'cache/metrics.jsonl').read_text())
            def history(folder):
                rows=[json.loads(line) for line in (folder/'history.jsonl').read_text().splitlines()]
                for row in rows:row.pop('seconds')
                return rows
            self.assertEqual(history(root/'memory'),history(root/'cache'))
            # Resume streams the cache: no teacher preparation and identical bits.
            resumed=self.run_native(root,'resumed',base+teacher+['--steps',1,'--resume',root/'cache/step-2.state.gguf','--out',root/'resumed'])
            self.assertNotIn('prepare_teacher',resumed.stdout)
            self.assertEqual(tensors(root/'resumed/step-3.state.gguf'),tensors(root/'memory/step-3.state.gguf'))
            # Any provenance mismatch is rejected before training.
            for index,value in [(teacher.index('--teacher-identity')+1,'other'),(teacher.index('--teacher-base-sha256')+1,'other')]:
                wrong=list(teacher);wrong[index]=value
                rejected=self.run_native(root,'rejected',base+wrong+['--steps',1,'--out',root/f'rejected-{index}'],ok=False)
                self.assertNotEqual(rejected.returncode,0);self.assertIn('different model/configuration',rejected.stderr)
            rejected=self.run_native(root,'rejected-slots',base+teacher+['--max-prefix',2,'--steps',1,'--out',root/'rejected-slots'],ok=False)
            self.assertIn('different model/configuration',rejected.stderr)

    def test_lora_streamed_teacher_matches_memory_and_resume(self):self.check()

    def test_dora_streamed_teacher_matches_memory_and_resume(self):self.check(True)


if __name__=='__main__':unittest.main()
