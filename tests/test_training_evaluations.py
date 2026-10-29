"""Live charts use completed recording-balanced metrics, never provisional selection."""
import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0,str(Path(__file__).parents[1]/'tools'))
from train_ar_run import evaluation_history,select_checkpoints


class EvaluationHistoryTest(unittest.TestCase):
    rows=[dict(split='validation',group=group) for group in ['a','a','b']]

    def test_live_ar_records_partial_append_and_recording_balance(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);native=root/'native-1';native.mkdir();path=native/'metrics.jsonl'
            zero=dict(step=0,heldout_per_sample=[1,3,7],retention_per_sample=[])
            hundred=dict(step=100,heldout_per_sample=[1,1,3],retention_per_sample=[])
            line=json.dumps(hundred)+'\n';path.write_text(json.dumps(zero)+'\n'+line[:20])
            history=evaluation_history(root,self.rows)
            self.assertEqual(len(history['evaluations']),1)
            self.assertEqual(history['best_so_far']['heldout_group_mean'],4.5)
            self.assertNotIn('selected',history)
            self.assertFalse((root/'selection.json').exists())
            with path.open('a') as stream:stream.write(line[20:])
            history=evaluation_history(root,self.rows)
            self.assertEqual([m['step'] for m in history['evaluations']],[0,100])
            self.assertEqual(history['best_so_far']['heldout_group_mean'],2)

    def test_numeric_resume_order_wins_for_same_evaluated_step(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            for segment,value in [('native-10',2),('native-2',4)]:
                native=root/segment;native.mkdir()
                (native/'metrics.jsonl').write_text(json.dumps(dict(step=100,heldout_per_sample=[value]*3,retention_per_sample=[]))+'\n')
                (native/'step-100.gguf').write_bytes(b'identical resumed checkpoint')
            self.assertEqual(evaluation_history(root,self.rows)['best_so_far']['heldout_group_mean'],2)
            select_checkpoints(root,self.rows,{})
            self.assertEqual(json.loads((root/'selection.json').read_text())['selected']['heldout_group_mean'],2)

    def test_nar_skips_optimizer_records_and_keeps_retention_separate(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);native=root/'native-1';native.mkdir()
            records=[dict(step=1,loss=.4),dict(step=100,heldout_eval=.8,heldout_per_sample=[1,3,7],retention_per_sample=[9])]
            (native/'history.jsonl').write_text(''.join(json.dumps(m)+'\n' for m in records))
            result=evaluation_history(root,self.rows+[dict(split='retention',group='neutral')],'nar')
            self.assertEqual(len(result['evaluations']),1)
            self.assertEqual(result['best_so_far']['heldout_group_mean'],4.5)
            self.assertEqual(result['best_so_far']['retention_group_mean'],9)

    def test_incomplete_or_nonfinite_metrics_are_not_charted(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);native=root/'native-1';native.mkdir();path=native/'metrics.jsonl'
            path.write_text(json.dumps(dict(step=100,heldout_per_sample=[1]))+'\n')
            with self.assertRaisesRegex(ValueError,'Metric/example count'):evaluation_history(root,self.rows)
            path.write_text(json.dumps(dict(step=100,heldout_per_sample=[float('nan')]*3))+'\n')
            with self.assertRaisesRegex(ValueError,'Nonfinite'):evaluation_history(root,self.rows)


if __name__=='__main__':unittest.main()
