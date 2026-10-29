import sys
from pathlib import Path
import tempfile
import unittest
import gguf
import numpy as np

sys.path.insert(0,str(Path(__file__).parents[1]/'tools'))
from train_nar_run import preflight_sample


class PreflightTest(unittest.TestCase):
    def test_first_training_sample_uses_maximum_capacity_including_holdouts(self):
        with tempfile.TemporaryDirectory() as directory:
            rows=[]
            for split,tokens in [('train',2),('train',4),('validation',8)]:
                path=Path(directory)/f'{tokens}.gguf';writer=gguf.GGUFWriter(str(path),'yue2_nar_sample')
                writer.add_uint32('yue2.training.ar_tokens',tokens)
                writer.add_tensor('clean',np.zeros(128,dtype=np.float32))
                for key in ['keys','values']:writer.add_tensor(key,np.zeros(tokens*128*8*28,dtype=np.float16))
                writer.write_header_to_file();writer.write_kv_data_to_file();writer.write_tensors_to_file();writer.close()
                rows.append(dict(split=split,cache=str(path)))
            first,maximum=preflight_sample(rows,lambda r:Path(r['cache']),2)
            self.assertEqual(first,Path(rows[0]['cache']))
            self.assertEqual(maximum,8)
            with self.assertRaisesRegex(ValueError,'shape'):preflight_sample(rows,lambda r:Path(r['cache']),3)


if __name__=='__main__':unittest.main()
