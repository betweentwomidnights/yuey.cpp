"""Optional real-model production integration, with torch imports prohibited."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
import numpy as np
import soundfile as sf

ROOT=Path(__file__).resolve().parents[1]
BINARY=os.environ.get('YUE2_VAE_ENCODER_TEST_BINARY')
MODEL=os.environ.get('YUE2_VAE_ENCODER_TEST_MODEL')
DEVICE=os.environ.get('YUE2_VAE_ENCODER_TEST_DEVICE','cuda')

@unittest.skipUnless(BINARY and MODEL,'Set native VAE binary/model paths for integration')
class NativeTargetsTest(unittest.TestCase):
    def test_torch_free_preparation_resume_and_profile_guard(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);audio=root/'source.wav';rate=48000
            t=np.arange(rate*3)/rate
            sf.write(audio,np.stack([.1*np.sin(440*2*np.pi*t),.07*np.sin(311*2*np.pi*t)],axis=1),rate,subtype='FLOAT')
            sha=hashlib.sha256(audio.read_bytes()).hexdigest()
            dataset=root/'dataset.json';dataset.write_text(json.dumps(dict(completed=True,rows=[dict(source=str(audio),source_sha256=sha,track='source.wav',start_frame=0,frames=50,split='train')])))
            out=root/'targets';args=['--dataset',str(dataset),'--vae',MODEL,'--binary',BINARY,'--out',str(out),'--device',DEVICE]
            code='''import importlib.abc,sys,runpy,json
class Guard(importlib.abc.MetaPathFinder):
 def find_spec(self,fullname,path=None,target=None):
  if fullname=='torch' or fullname.startswith('torch.'):raise RuntimeError('Torch imported by production preparation')
sys.meta_path.insert(0,Guard());sys.path.insert(0,'tools');sys.argv=['prepare_nar_targets.py',*json.loads(sys.argv[1])]
runpy.run_path('tools/prepare_nar_targets.py',run_name='__main__')
assert 'torch' not in sys.modules
'''
            def run(extra=()):
                return subprocess.run([sys.executable,'-c',code,json.dumps(args+list(extra))],cwd=ROOT,capture_output=True,text=True,timeout=120)
            first=run();self.assertEqual(first.returncode,0,first.stderr)
            manifest=json.loads((out/'index.json').read_text());self.assertTrue(manifest['completed'])
            target=Path(manifest['rows'][0]['means']);before=target.read_bytes()
            self.assertEqual(len(before),50*64*4)
            resumed=run();self.assertEqual(resumed.returncode,0,resumed.stderr);self.assertEqual(before,target.read_bytes())
            # Missing provenance never silently blesses an old payload.
            metadata=target.with_name('target.json');metadata.rename(metadata.with_name('previous.json'))
            recovered=run();self.assertEqual(recovered.returncode,0,recovered.stderr)
            self.assertEqual(before,target.read_bytes());self.assertTrue(list(target.parent.glob('means.unregistered-*.f32')))
            changed=run(['--core-frames','64']);self.assertNotEqual(changed.returncode,0)
            self.assertIn('profile changed',changed.stderr);self.assertEqual(before,target.read_bytes())

if __name__=='__main__':unittest.main()
