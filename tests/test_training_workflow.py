import json
from pathlib import Path
import sys
import tempfile
import os
from unittest.mock import patch
import unittest
import numpy as np
import soundfile as sf

sys.path.insert(0,str(Path(__file__).parents[1]/'tools'))
from training_workflow import settings,inventory,wait_for_owner,process_identity,Workflow,ar_resource_estimate


class WorkflowTest(unittest.TestCase):
    def test_new_recipe_pins_faster_validation_without_changing_legacy_requests(self):
        self.assertNotIn('evaluation_scope',settings(dict(id='old-run')))
        self.assertEqual(settings(dict(id='new-run',recipe_version=2))['evaluation_scope'],'heldout')
        self.assertEqual(settings(dict(id='new-run',recipe_version=2,evaluation_scope='full'))['evaluation_scope'],'full')
        with self.assertRaisesRegex(ValueError,'Evaluation scope'):
            settings(dict(id='new-run',recipe_version=2,evaluation_scope='none'))
        self.assertNotIn('cache_fingerprint',settings(dict(id='version-two',recipe_version=2)))
        self.assertEqual(settings(dict(id='version-three',recipe_version=3))['cache_fingerprint'],'bytes-v2')
        self.assertNotIn('teacher_cache',settings(dict(id='version-three',recipe_version=3)))
        four=settings(dict(id='version-four',recipe_version=4))
        self.assertEqual((four['teacher_cache'],four['cache_fingerprint'],four['evaluation_scope']),('f32','bytes-v2','heldout'))
        with self.assertRaisesRegex(ValueError,'recipe version'):settings(dict(id='new-run',recipe_version=5))

    def test_teacher_estimates_match_measured_koan_corpus(self):
        # 157 training and 38 validation 30-second windows, measured in
        # docs/ar-teacher-cache-2026-10-28.md: 15.53 GiB versus 3.09 GiB peak.
        rows=[dict(split='train',frames=750,prefix_tokens=1000)]*157+[dict(split='validation',frames=750,prefix_tokens=1000)]*38
        old=ar_resource_estimate(rows,settings(dict(id='v3',recipe_version=3)),'cuda')
        new=ar_resource_estimate(rows,settings(dict(id='v4',recipe_version=4)),'cuda')
        self.assertAlmostEqual(old['estimated_ar_host_ram_gib'],15.5,delta=.1)
        self.assertEqual(old['estimated_teacher_disk_gib'],0)
        self.assertAlmostEqual(new['estimated_ar_host_ram_gib'],3.1,delta=.05)
        self.assertAlmostEqual(new['estimated_teacher_disk_gib'],14.38,delta=.02)
        # Streamed host RAM does not grow with the corpus; disk does.
        double=ar_resource_estimate(rows*2,settings(dict(id='v4',recipe_version=4)),'cuda')
        self.assertEqual(double['estimated_ar_host_ram_gib'],new['estimated_ar_host_ram_gib'])
        self.assertAlmostEqual(double['estimated_teacher_disk_gib'],2*new['estimated_teacher_disk_gib'])
        cpu=ar_resource_estimate(rows,settings(dict(id='v4',recipe_version=4)),'cpu',8*2**30)
        self.assertAlmostEqual(cpu['estimated_ar_host_ram_gib'],11.1,delta=.05)
        no_kl=ar_resource_estimate(rows,settings(dict(id='v4',recipe_version=4,kl_weight=0)),'cuda')
        self.assertEqual(no_kl['estimated_teacher_disk_gib'],0)

    def test_resume_does_not_display_previous_terminal_progress(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);run=root/'run';run.mkdir()
            (run/'runtime.json').write_text(json.dumps(dict(workspace=str(root))))
            (run/'request.json').write_text(json.dumps(dict(id='fixture')))
            progress=root/'progress.json';progress.write_text(json.dumps(dict(status='stopped',step=6)))
            class Finished:
                pid=123;returncode=0
                def __init__(self,*args,**kwargs):pass
                def poll(self):return 0
                def wait(self):return 0
            workflow=Workflow(run)
            with patch('training_workflow.subprocess.Popen',Finished):
                workflow.command('fixture.py',[],'ar-fit',progress)
            self.assertEqual(workflow.state['progress'],{})

    def test_fast_child_terminal_progress_is_observed(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);run=root/'run';run.mkdir()
            (run/'runtime.json').write_text(json.dumps(dict(workspace=str(root))))
            (run/'request.json').write_text(json.dumps(dict(id='fixture')))
            progress=root/'progress.json'
            class Finished:
                pid=123;returncode=0
                def __init__(self,*args,**kwargs):progress.write_text(json.dumps(dict(current=7,total=7)))
                def poll(self):return 0
                def wait(self):return 0
            workflow=Workflow(run)
            with patch('training_workflow.subprocess.Popen',Finished):
                workflow.command('fixture.py',[],'fixture',progress)
            self.assertEqual(workflow.state['progress'],dict(current=7,total=7))
    @unittest.skipUnless(sys.platform in ['win32','linux'],'Native owner identity uses Windows creation time or Linux procfs')
    def test_worker_requires_committed_pid_and_creation_identity(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);owner=root/'owner.json'
            owner.write_text(json.dumps(dict(pid=os.getpid(),identity=process_identity())))
            wait_for_owner(root,.01)
            self.assertEqual(process_identity(),process_identity(os.getpid()))
            if os.name=='nt':
                owner.write_text(json.dumps(dict(pid=os.getppid(),identity=process_identity(os.getppid()))))
                wait_for_owner(root,.01)
            owner.write_text(json.dumps(dict(pid=os.getpid(),identity='stale')))
            with self.assertRaisesRegex(RuntimeError,'not committed'):wait_for_owner(root,.01)
    def test_rejects_unsafe_budgets_and_nonfinite_hyperparameters(self):
        for field,value in [('ar_steps',0),('nar_steps',-1),('rank',2.5),('seed',True),
                            ('kl_weight',float('nan')),('ar_lr',float('inf')),('window_seconds',61),
                            ('family','other'),('trigger','two words')]:
            with self.subTest(field=field,value=value),self.assertRaises(ValueError):
                settings(dict(id='style',**{field:value}))

    def test_recording_holdout_is_deterministic_and_source_bytes_are_read_only(self):
        with tempfile.TemporaryDirectory() as directory:
            folder=Path(directory)
            for i in range(8):
                sf.write(folder/f'{i}.wav',np.sin(np.arange(16000)*(i+1)*.01),8000,subtype='FLOAT')
                (folder/f'{i}.txt').write_text('synthwave, 88 bpm, D minor')
            before={p.name:p.read_bytes() for p in folder.iterdir()}
            a=inventory(folder,2,123);b=inventory(folder,2,123)
            self.assertEqual(a,b);self.assertEqual(sum(r['split']=='validation' for r in a),1)
            self.assertEqual(before,{p.name:p.read_bytes() for p in folder.iterdir()})
            # Two filenames cannot cause the same recording to straddle splits.
            (folder/'duplicate.wav').write_bytes((folder/'0.wav').read_bytes())
            (folder/'duplicate.txt').write_text('synthwave, 88 bpm, D minor')
            with self.assertRaisesRegex(ValueError,'Duplicate recording'):inventory(folder,2,123)

    def test_missing_captions_and_short_audio_fail_before_model_loading(self):
        with tempfile.TemporaryDirectory() as directory:
            folder=Path(directory);audio=folder/'one.wav';sf.write(audio,np.zeros(8000),8000)
            with self.assertRaisesRegex(ValueError,'Missing caption'):inventory(folder,2,123)
            audio.with_suffix('.txt').write_text('synthwave, 88 bpm, D minor')
            with self.assertRaisesRegex(ValueError,'shorter'):inventory(folder,2,123)


if __name__=='__main__':unittest.main()
