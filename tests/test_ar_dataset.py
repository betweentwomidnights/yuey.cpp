import importlib.util
import os
from pathlib import Path
import struct
import subprocess
import tempfile
import unittest

spec=importlib.util.spec_from_file_location('ar_dataset',Path(__file__).parents[1]/'tools/ar_dataset.py')
dataset=importlib.util.module_from_spec(spec);spec.loader.exec_module(dataset)


def midi_notes(path):
    """Independent SMF timeline reader for the native export roundtrip."""
    data=Path(path).read_bytes();assert data[:4]==b'MThd'
    size=struct.unpack('>I',data[4:8])[0];_,count,division=struct.unpack('>HHH',data[8:14]);at=8+size
    notes=[];tempo=500000
    for track in range(count):
        assert data[at:at+4]==b'MTrk';size=struct.unpack('>I',data[at+4:at+8])[0];end=at+8+size;at+=8;tick=0;running=None;active={}
        def variable():
            nonlocal at
            value=0
            while True:
                byte=data[at];at+=1;value=(value<<7)|(byte&127)
                if byte<128:return value
        while at<end:
            tick+=variable();status=data[at]
            if status>=128:at+=1;running=status
            else:status=running
            if status==255:
                kind=data[at];at+=1;n=variable();payload=data[at:at+n];at+=n
                if kind==81:tempo=int.from_bytes(payload,'big')
            elif status in [240,247]:at+=variable()
            else:
                kind=status&240;n=1 if kind in [192,208] else 2;payload=data[at:at+n];at+=n
                if kind==144 and payload[1]>0:active[(status&15,payload[0])]=tick
                elif kind==128 or (kind==144 and payload[1]==0):
                    start=active.pop((status&15,payload[0]),None)
                    if start is not None:notes.append((payload[0],start/division*tempo/1e6,tick/division*tempo/1e6))
    return sorted(notes,key=lambda note:note[1])


class DatasetTest(unittest.TestCase):
    def test_caption_metadata_keeps_style_and_unicode_key(self):
        result=dataset.caption_metadata('neurofunk, dark electronica, 88 bpm, B\u266d major')
        self.assertEqual(result,dict(bpm=88.,key='Bb',style='neurofunk, dark electronica'))

    def test_full_real_windows_include_short_recordings_without_padding(self):
        starts=dataset.window_starts(1129,750,85)
        self.assertGreaterEqual(len(starts),2)
        self.assertTrue(all(0<=start<=1129-750 for start in starts))
        self.assertEqual(starts[0],0)
        self.assertGreater(starts[-1]+750,1100)
        with self.assertRaises(ValueError):dataset.window_starts(749,750,85)

    def test_chord_inversions_match_native_chord_spelling(self):
        self.assertEqual(dataset.chord_name('E:maj/3'),'E/G#')
        self.assertEqual(dataset.chord_name('B:min7/b7'),'Bm7/A')
        self.assertEqual(dataset.chord_name('G:min/2'),'Gm/A')

    def test_native_midi_roundtrip_preserves_absolute_notes_and_triplets(self):
        binary=Path(os.environ.get('YUE2_SCORE_TEST_BINARY',''))
        if not binary.is_file():self.skipTest('Set YUE2_SCORE_TEST_BINARY for native roundtrip')
        events=[dict(time_seconds=t,meter_numerator=4 if i==0 else 0,meter_denominator=4 if i==0 else 0,chord='',notes=[dict(pitch=p,track=1,end_time_seconds=end)]) for i,(t,end,p) in enumerate([(0,1/3,59),(1/3,.5,64),(.5,1,63),(1.5,2.5,71)])]
        for key in ['Fm','D','G#m']:
            abc,stats=dataset.score_from_events(events,3,120,key)
            self.assertIn('Q:1/4=120',abc)
            self.assertIn('K:'+key,abc)
            self.assertLessEqual(stats['max_rounding_seconds'],stats['max_expected_grid_error_seconds'])
            with tempfile.TemporaryDirectory() as temp:
                folder=Path(temp);source=folder/'input.abc';source.write_text(abc)
                result=subprocess.run([str(binary),'--input',str(source),'--output',str(folder/'used.abc'),'--midi-dir',str(folder/'midi')],capture_output=True,text=True)
                self.assertEqual(result.returncode,0,result.stderr)
                notes=midi_notes(folder/'midi/instrumental.mid')
                self.assertEqual([n[0] for n in notes],[59,64,63,71])
                for actual,expected in zip(notes,[(59,0,1/3),(64,1/3,.5),(63,.5,1),(71,1.5,2.5)]):
                    self.assertAlmostEqual(actual[1],expected[1],places=5)
                    self.assertAlmostEqual(actual[2],expected[2],places=5)


if __name__=='__main__':unittest.main()
