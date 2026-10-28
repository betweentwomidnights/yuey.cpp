import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0,str(Path(__file__).parents[1]/'tools'))
from train_nar_run import NativeNarProgress


class NarProgressTest(unittest.TestCase):
    def test_validation_and_initial_evaluation_do_not_count_as_optimizer_updates(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'native.log';path.write_text('')
            progress=NativeNarProgress(path,400)
            def event(stage,current,total,step=0):
                with path.open('a') as stream:
                    stream.write('TRAIN_EVENT '+json.dumps(dict(stage=stage,current=current,total=total,step=step))+'\n')
                return progress.read()
            result=event('cache-validation',127,166)
            self.assertEqual(result['step'],0)
            self.assertEqual(result['phase_current'],127)
            self.assertEqual(result['phase_unit'],'files')
            result=event('model-loading',0,1)
            self.assertEqual(result['activity'],'Loading native NAR model')
            self.assertNotIn('phase_current',result)
            result=event('evaluation',61,166)
            self.assertEqual(result['activity'],'Initial evaluation')
            self.assertEqual(result['phase_current'],61)
            self.assertEqual(result['phase_unit'],'windows')
            self.assertEqual(result['step'],0)
            with path.open('a') as stream:stream.write('{"step":1,"loss":0.8}\n')
            result=progress.read()
            self.assertEqual(result['step'],1)
            self.assertEqual(result['activity'],'Training')
            self.assertNotIn('phase_current',result)

    def test_split_event_resume_and_final_evaluation(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'native.log';path.write_text('TRAIN_EVENT {"stage":"cache-validation",')
            progress=NativeNarProgress(path,400,200)
            self.assertEqual(progress.read()['step'],200)
            with path.open('a') as stream:stream.write('"current":1,"total":166,"step":0}\n')
            self.assertEqual(progress.read()['step'],200)
            with path.open('a') as stream:
                stream.write('TRAIN_EVENT {"stage":"evaluation","current":28,"total":166,"step":400}\n')
            result=progress.read()
            self.assertEqual(result['step'],400)
            self.assertEqual(result['activity'],'Final evaluation')
            self.assertEqual(result['phase_current'],28)
            with path.open('a') as stream:
                stream.write('TRAIN_EVENT {"stage":"checkpoint-save","current":0,"total":1,"step":400}\n')
            self.assertEqual(progress.read()['activity'],'Saving checkpoint')

    def test_old_native_binary_banner_has_a_useful_fallback(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'native.log';path.write_text('[yue2:train] 750 frames\n')
            progress=NativeNarProgress(path,400)
            self.assertIn('initial evaluation',progress.read()['activity'].lower())
            self.assertFalse(progress.events)
            progress=NativeNarProgress(path,400,1)
            self.assertEqual(progress.read()['activity'],'Loading native optimizer state')


if __name__=='__main__':unittest.main()
