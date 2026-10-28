"""Checkpoint retention and cache release keep everything selection, resume
and publishing need, and nothing else."""
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0,str(Path(__file__).parents[1]/'tools'))
from train_ar_run import prune_checkpoints,select_checkpoints
from training_workflow import Workflow,checkpoint_allowance,conditioning_bytes,settings


def metric(step,value):
    return json.dumps(dict(step=step,heldout_per_sample=[value],retention_per_sample=[]))+'\n'


def checkpoint(folder,step,state=True):
    folder.mkdir(parents=True,exist_ok=True)
    (folder/f'step-{step}.gguf').write_bytes(b'adapter %d'%step)
    if state:(folder/f'step-{step}.state.gguf').write_bytes(b'state %d'%step)


def names(out):
    return sorted(str(p.relative_to(out)).replace('\\','/') for p in out.glob('native*/step-*.gguf'))


ROWS=[dict(split='validation',group='heldout')]


class CheckpointRetentionTest(unittest.TestCase):
    def test_keeps_zero_best_and_latest_across_resumed_segments(self):
        with tempfile.TemporaryDirectory() as directory:
            out=Path(directory);old=out/'native-1';new=out/'native-2'
            checkpoint(old,0,state=False)
            for step in [100,200]:checkpoint(old,step)
            for step in [300,400]:checkpoint(new,step)
            (old/'metrics.jsonl').write_text(metric(0,5)+metric(100,4.5)+metric(200,4))
            # Resume re-evaluates the loaded step-200 checkpoint in the new segment.
            (new/'metrics.jsonl').write_text(metric(200,4)+metric(300,4.2)+metric(400,4.3))
            released=prune_checkpoints(out,ROWS)
            self.assertEqual(names(out),['native-1/step-0.gguf','native-1/step-200.gguf','native-1/step-200.state.gguf',
                                         'native-2/step-400.gguf','native-2/step-400.state.gguf'])
            self.assertGreater(released,0)
            select_checkpoints(out,ROWS,{})
            self.assertEqual(json.loads((out/'selection.json').read_text())['adapter'],str(old/'step-200.gguf'))

    def test_unevaluated_and_in_flight_checkpoints_survive(self):
        with tempfile.TemporaryDirectory() as directory:
            out=Path(directory);native=out/'native-1'
            for step in [10,20,30]:checkpoint(native,step)
            # Step 20 has no evaluation record yet (e.g. an off-grid stop).
            # The adapter for step 40 is visible but its state is still being written.
            (native/'step-40.gguf').write_bytes(b'adapter 40')
            (native/'step-40.state.gguf.partial').write_bytes(b'partial')
            (native/'metrics.jsonl').write_text(metric(10,3)+metric(30,4))
            prune_checkpoints(out,ROWS)
            self.assertEqual(names(out),['native-1/step-10.gguf','native-1/step-10.state.gguf',
                                         'native-1/step-20.gguf','native-1/step-20.state.gguf',
                                         'native-1/step-30.gguf','native-1/step-30.state.gguf',
                                         'native-1/step-40.gguf'])
            self.assertTrue((native/'step-40.state.gguf.partial').exists())

    def test_nar_reads_history_and_ignores_optimizer_records(self):
        with tempfile.TemporaryDirectory() as directory:
            out=Path(directory);native=out/'native-1'
            for step in [2,4,6]:checkpoint(native,step)
            records=[json.dumps(dict(step=1,loss=1.0)),json.dumps(dict(step=2,heldout_eval=True,heldout_per_sample=[1.2])),
                     json.dumps(dict(step=4,heldout_eval=True,heldout_per_sample=[1.1])),
                     json.dumps(dict(step=6,heldout_eval=True,heldout_per_sample=[1.3]))]
            (native/'history.jsonl').write_text('\n'.join(records)+'\n')
            prune_checkpoints(out,ROWS,'nar')
            self.assertEqual(names(out),['native-1/step-4.gguf','native-1/step-4.state.gguf',
                                         'native-1/step-6.gguf','native-1/step-6.state.gguf'])

    def test_locked_state_keeps_its_adapter_until_a_later_pass(self):
        with tempfile.TemporaryDirectory() as directory:
            out=Path(directory);native=out/'native-1'
            for step in [2,4,6]:checkpoint(native,step)
            (native/'metrics.jsonl').write_text(metric(2,3)+metric(4,4)+metric(6,5))
            unlink=Path.unlink
            def locked(path,*args,**kwargs):
                # Windows refuses to delete a file another reader has mapped.
                if path.name=='step-4.state.gguf':raise PermissionError(32,'in use')
                return unlink(path,*args,**kwargs)
            with patch.object(Path,'unlink',locked):prune_checkpoints(out,ROWS)
            self.assertIn('native-1/step-4.gguf',names(out))
            prune_checkpoints(out,ROWS)
            self.assertEqual(names(out),['native-1/step-2.gguf','native-1/step-2.state.gguf',
                                         'native-1/step-6.gguf','native-1/step-6.state.gguf'])

    def test_no_evaluations_or_states_removes_nothing(self):
        with tempfile.TemporaryDirectory() as directory:
            out=Path(directory);checkpoint(out/'native-1',5)
            self.assertEqual(prune_checkpoints(out,ROWS),0)
            (out/'native-1'/'metrics.jsonl').write_text(metric(5,1))
            self.assertEqual(prune_checkpoints(out,ROWS),0)
            self.assertEqual(len(names(out)),2)


class CacheReleaseTest(unittest.TestCase):
    def run_fixture(self,root,**request):
        run=root/'run';run.mkdir()
        (run/'runtime.json').write_text(json.dumps(dict(workspace=str(root))))
        (run/'request.json').write_text(json.dumps(dict(id='fixture',family='lora',rank=4,**request)))
        return run

    def test_conditioning_disk_by_storage_mode(self):
        # Shaped like the 38-recording laptop corpus: 138 training, 28 heldout windows.
        rows=[dict(split='train',frames=750,prefix_tokens=1131)]*138+[dict(split='validation',frames=750,prefix_tokens=1131)]*28
        cache,rolling=(conditioning_bytes(rows,mode,'heldout')/2**30 for mode in ['cache','rolling'])
        self.assertAlmostEqual(cache,33.4,delta=.05)  # every prefix at the maximum
        self.assertAlmostEqual(rolling,(25+28)*conditioning_bytes(rows[:1])/2**30,delta=1e-6)
        self.assertLess(rolling,cache/3)
        self.assertEqual(conditioning_bytes(rows,'resident'),0)
        # Full-scope evaluation needs every window, so rolling saves nothing.
        self.assertEqual(conditioning_bytes(rows,'rolling','full'),conditioning_bytes(rows))
        self.assertEqual(conditioning_bytes(rows[:3],'rolling','heldout'),conditioning_bytes(rows[:3]))

    def test_keep_caches_is_a_validated_boolean_defaulting_to_release(self):
        self.assertFalse(settings(dict(id='run'))['keep_caches'])
        self.assertTrue(settings(dict(id='run',keep_caches=True))['keep_caches'])
        with self.assertRaisesRegex(ValueError,'keep_caches'):settings(dict(id='run',keep_caches='yes'))

    def test_release_keeps_selected_adapters_and_blocks_further_training(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);run=self.run_fixture(root)
            lora=run/'data/lora';nar=run/'nar-run'
            checkpoint(lora/'native-1',0,state=False);checkpoint(lora/'native-1',20);checkpoint(lora/'native-1',40)
            (lora/'probe').mkdir();(lora/'probe/step-0.gguf').write_bytes(b'probe')
            (lora/'probe-diagnostics.gguf').write_bytes(b'diagnostics');(lora/'prompts.json').write_text('{}')
            checkpoint(nar/'native-1',30);checkpoint(nar/'native-1',60)
            cache=run/'nar-conditioning/window-0';cache.mkdir(parents=True);(cache/'sample.gguf').write_bytes(b'kv')
            (run/'nar-conditioning/dataset.json').write_text('{}')
            example=run/'data/examples/window-0';example.mkdir(parents=True)
            (example/'source.wav').write_bytes(b'audio');(example/'codes.i32').write_bytes(b'codes')
            workflow=Workflow(run)
            workflow.state.update(ar_selection=dict(adapter=str(lora/'native-1/step-20.gguf')),
                                  nar_selection=dict(adapter=str(nar/'native-1/step-60.gguf')))
            workflow.release_caches()
            left=sorted(str(p.relative_to(run)).replace('\\','/') for p in run.rglob('*') if p.is_file())
            self.assertEqual(left,['data/examples/window-0/codes.i32','data/lora/native-1/step-20.gguf','data/lora/prompts.json',
                                   'nar-conditioning/dataset.json','nar-run/native-1/step-60.gguf','request.json','runtime.json','state.json'])
            self.assertGreater(workflow.state['caches_released']['gib'],0)
            (run/'vae-targets').mkdir();(run/'vae-targets/index.json').write_text('{}')
            with self.assertRaisesRegex(ValueError,'released its training caches'):Workflow(run).train()

    def test_publish_honours_keep_caches(self):
        for keep,expected in [(False,1),(True,0)]:
            with tempfile.TemporaryDirectory() as directory:
                workflow=Workflow(self.run_fixture(Path(directory),keep_caches=keep))
                workflow.state['status']='review'
                with patch('publish_training_adapter.publish',return_value={}), \
                     patch.object(Workflow,'release_caches') as release:
                    workflow.publish()
                self.assertEqual(release.call_count,expected)

    def test_disk_check_counts_remaining_work_only(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);run=self.run_fixture(root,recipe_version=4)
            rows=[dict(split='train',frames=750,prefix_tokens=1000)]*10+[dict(split='validation',frames=750,prefix_tokens=1000)]*2
            dataset=run/'data/dataset.json';dataset.parent.mkdir(parents=True)
            dataset.write_text(json.dumps(dict(rows=rows)))
            (run/'ar-job.json').write_text(json.dumps(dict(training_dataset=str(dataset))))
            workflow=Workflow(run);ar=run/'data/lora';nar=run/'nar-conditioning'
            class Usage:free=0
            with patch('training_workflow.shutil.disk_usage',return_value=Usage):
                with self.assertRaisesRegex(ValueError,'GiB free on the run drive'):workflow.check_disk(ar,nar)
            # With AR selected and conditioning complete only NAR checkpoints remain.
            ar.mkdir(parents=True);(ar/'selection.json').write_text('{}')
            nar.mkdir();(nar/'dataset.json').write_text(json.dumps(dict(completed=True)))
            Usage.free=checkpoint_allowance(workflow.options,1)
            with patch('training_workflow.shutil.disk_usage',return_value=Usage):workflow.check_disk(ar,nar)

    def test_rolling_regenerates_a_redrawn_window_once(self):
        import hashlib
        import prepare_nar_dataset
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);manifest=root/'dataset.json'
            manifest.write_text(json.dumps(dict(identity=dict(device='cuda',threads=6,kv_type='f16',prefill_batch=0))))
            facts=dict(cache_bytes='5',cache_fnv='0',cache_tokens='1',cache_frames='1',cache_half=True)
            row=dict(prefix='p',codes='c',means='m',cache=str(root/'a/sample.gguf'),cache_identity='i',
                     cache_sha256=hashlib.sha256(b'cache').hexdigest(),**facts)
            def batch(command,workspace,log,on_event):
                plan=json.loads((root/'materialize.json').read_text())
                self.assertEqual(len(plan),1)
                path=Path(plan[0]['cache-out']);path.parent.mkdir(exist_ok=True);path.write_bytes(b'cache')
                on_event(dict(path=str(path),bytes='5',fnv='0',tokens='1',frames='1',half=True))
            # The sampler drew the same window twice in one segment.
            with patch('prepare_nar_dataset.run_batch',batch):
                self.assertEqual(prepare_nar_dataset.materialize(manifest,[row,dict(row)],'yue2-train','b','c','a'),1)


if __name__=='__main__':unittest.main()
