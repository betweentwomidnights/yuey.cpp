import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0,str(Path(__file__).parents[1]/'tools'))
from train_ar_run import NativeProgress,prepare_teacher_cache,verify_teacher_cache,teacher_identity

# Stands in for `yue2-ar-train --prepare-teacher-only`: writes missing files
# only, as the native writer does, and records how many it wrote.
FAKE='''
import sys
from pathlib import Path
args=sys.argv[1:];folder=Path(args[args.index('--teacher-cache')+1]);count=int(args[args.index('--count')+1])
assert args[-1]=='--prepare-teacher-only' and args.count('--teacher-cache')==1
assert args[args.index('--teacher-base-sha256')+1]=='base' and args[args.index('--teacher-companion-sha256')+1]=='companion'
assert len(args[args.index('--teacher-identity')+1])==64
written=0
for i in range(count):
    print(f'prepare_teacher={i+1}/{count}',flush=True)
    path=folder/f'teacher-{i:05d}.gguf'
    if not path.exists():path.write_bytes(bytes([i])*64);written+=1
(folder.parent/'written.txt').write_text(str(written))
'''


class TeacherIndexTest(unittest.TestCase):
    def setUp(self):
        self.directory=tempfile.TemporaryDirectory();self.root=Path(self.directory.name)
        (self.root/'fake.py').write_text(FAKE)
        self.rows=[dict(split='train',prefix_sha256=f'p{i}',codes_sha256=f'c{i}') for i in range(3)]
        identity=dict(base_sha256='base',companion_sha256='companion',dataset_sha256='data',runtime=dict(x='y'))
        self.resolved=teacher_identity(identity,dict(device='cuda'),dict(frames=750),1131,'f32')
        self.folder=self.root/'teacher'

    def tearDown(self):self.directory.cleanup()

    def prepare(self,resolved=None):
        teacher=self.folder;teacher.mkdir(exist_ok=True)
        cmd=[sys.executable,str(self.root/'fake.py'),'--count',str(len(self.rows))]
        return prepare_teacher_cache(teacher,cmd,self.root,self.root,'lora',resolved or self.resolved,self.rows,self.root/'stop.request')

    def test_index_records_hashes_and_detects_changes(self):
        sha=self.prepare()
        index=verify_teacher_cache(self.folder,sha,self.rows)
        self.assertEqual([f['path'] for f in index['files']],[f'teacher-{i:05d}.gguf' for i in range(3)])
        self.assertEqual((index['base_sha256'],index['companion_sha256'],index['ar_slots'],index['frames']),('base','companion',1131,750))
        self.assertEqual(index['bytes'],3*64)
        # A completed index is reused without invoking the native writer again.
        (self.root/'written.txt').unlink();self.assertEqual(self.prepare(),sha)
        self.assertFalse((self.root/'written.txt').exists())
        (self.folder/'teacher-00001.gguf').write_bytes(b'changed')
        with self.assertRaisesRegex(ValueError,'bytes changed'):verify_teacher_cache(self.folder,sha,self.rows)
        with self.assertRaisesRegex(ValueError,'does not match'):verify_teacher_cache(self.folder,'other',self.rows)
        changed=[dict(r) for r in self.rows];changed[0]['codes_sha256']='other'
        with self.assertRaisesRegex(ValueError,'provenance changed'):verify_teacher_cache(self.folder,sha,changed)
        with self.assertRaisesRegex(ValueError,'window count'):verify_teacher_cache(self.folder,sha,self.rows[:2])

    def test_interrupted_preparation_resumes_and_identity_is_immutable(self):
        self.folder.mkdir();(self.folder/'teacher-00000.gguf').write_bytes(bytes([0])*64)
        self.prepare()
        self.assertEqual((self.root/'written.txt').read_text(),'2')
        other=dict(self.resolved,device='cpu')
        with self.assertRaisesRegex(ValueError,'identity changed'):self.prepare(other)
        with self.assertRaisesRegex(ValueError,'Unsupported'):teacher_identity({},{},{},1,'f16')

    def test_unexpected_files_are_rejected(self):
        self.folder.mkdir();(self.folder/'teacher-00009.gguf').write_bytes(b'stale')
        with self.assertRaisesRegex(ValueError,'Unexpected AR teacher'):self.prepare()
        self.assertFalse((self.folder/'index.json').exists())

    def test_progress_reports_streamed_teacher_validation(self):
        log=self.root/'native.log';log.write_text('teacher_cache=4/157\n')
        state=NativeProgress(log,200).read()
        self.assertEqual((state['activity'],state['teacher_current'],state['teacher_total'],state['step']),
                         ('Validating frozen teacher cache',4,157,200))


if __name__=='__main__':unittest.main()
