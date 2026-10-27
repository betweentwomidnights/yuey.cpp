"""Scientific dataset controls: context, alignment and independent target cores."""
import importlib.util
from pathlib import Path
import sys
import unittest

TOOLS = Path(__file__).parents[1]/'tools'
sys.path.insert(0,str(TOOLS))
spec=importlib.util.spec_from_file_location('prepare_nar_experiment',TOOLS/'prepare_nar_experiment.py')
prepare=importlib.util.module_from_spec(spec)
spec.loader.exec_module(prepare)


class WindowTest(unittest.TestCase):
    def test_shortest_training_track_keeps_context_and_disjoint_cores(self):
        duration=72.62
        starts=prepare.fractional_window_starts(duration,256,[.1,.35,.6,.85])
        self.assertEqual(len(starts),4)
        self.assertGreaterEqual(starts[0],50)
        self.assertLessEqual(starts[-1]+256+50,int(duration*25))
        self.assertTrue(all(b-a>=256 for a,b in zip(starts,starts[1:])))
        self.assertTrue(all(isinstance(s,int) for s in starts))

    def test_endpoint_windows_retain_two_second_halos(self):
        starts=prepare.fractional_window_starts(100,256,[0,1])
        self.assertEqual(starts,[50,2194])

    def test_nearby_fractions_cannot_masquerade_as_independent_audio(self):
        for fractions in ([.1,.11],[.1,.1]):
            with self.assertRaises(ValueError):
                prepare.fractional_window_starts(100,256,fractions)

    def test_invalid_fraction_and_too_short_track_rejected(self):
        for duration,fractions in [(10,[.1]),(100,[float('nan')]),(100,[-.1]),(100,[1.1])]:
            with self.assertRaises(ValueError):
                prepare.fractional_window_starts(duration,256,fractions)


if __name__=='__main__': unittest.main()
