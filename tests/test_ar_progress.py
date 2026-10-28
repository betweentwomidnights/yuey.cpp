import sys
import json
import tempfile
from pathlib import Path
import unittest

sys.path.insert(0,str(Path(__file__).parents[1]/'tools'))
from train_ar_run import NativeProgress,select_checkpoints


class ArProgressTest(unittest.TestCase):
    def test_resumed_initial_evaluation_selects_checkpoint_from_original_segment(self):
        with tempfile.TemporaryDirectory() as directory:
            out=Path(directory);old=out/'native-1';new=out/'native-2';old.mkdir();new.mkdir()
            (old/'step-5.gguf').write_bytes(b'original saved adapter')
            (old/'step-5.state.gguf').write_bytes(b'original optimizer')
            (new/'step-6.gguf').write_bytes(b'next adapter')
            def metric(step,value):
                return json.dumps(dict(step=step,heldout_per_sample=[value],retention_per_sample=[]))+'\n'
            (old/'metrics.jsonl').write_text(metric(0,5))
            (new/'metrics.jsonl').write_text(metric(5,4)+metric(6,4.1))
            rows=[dict(split='validation',group='heldout')]
            select_checkpoints(out,rows,{})
            selected=json.loads((out/'selection.json').read_text())
            self.assertEqual(selected['adapter'],str(old/'step-5.gguf'))
            self.assertEqual(selected['state'],str(old/'step-5.state.gguf'))
            (new/'step-5.gguf').write_bytes(b'a conflicting branch')
            with self.assertRaisesRegex(ValueError,'Conflicting adapters'):
                select_checkpoints(out,rows,{})

    def test_resume_warmup_keeps_saved_step_and_reports_teacher_work(self):
        with tempfile.TemporaryDirectory() as directory:
            log=Path(directory)/'native.log';log.write_text('')
            progress=NativeProgress(log,6)
            self.assertEqual(progress.read(),dict(activity='Loading native trainer',step=6))
            log.write_text('prepare_teacher=12/138\nprepare_teacher=13/')
            state=progress.read()
            self.assertEqual(state['teacher_current'],12)
            self.assertEqual(state['teacher_total'],138)
            self.assertEqual(state['step'],6)
            with log.open('a') as stream:stream.write('138\ninitial_loss=5.6 step=6\n')
            state=progress.read()
            self.assertEqual(state['teacher_current'],13)
            self.assertEqual(state['activity'],'Evaluating training and validation windows')
            with log.open('a') as stream:stream.write('step=7 loss=5.4 grad=0.4 seconds=3.6\n')
            state=progress.read()
            self.assertEqual(state['activity'],'Training')
            self.assertEqual(state['step'],7)

    def test_periodic_and_final_evaluation_are_identified(self):
        with tempfile.TemporaryDirectory() as directory:
            log=Path(directory)/'native.log';log.write_text('step=100 loss=5.0\n')
            progress=NativeProgress(log,eval_every=100,target_steps=150)
            self.assertEqual(progress.read()['activity'],'Evaluating checkpoint')
            with log.open('a') as stream:stream.write('step=101 loss=5.0\n')
            self.assertEqual(progress.read()['activity'],'Training')
            with log.open('a') as stream:stream.write('step=150 loss=5.0\n')
            self.assertEqual(progress.read()['activity'],'Evaluating checkpoint')


if __name__=='__main__':unittest.main()
