"""Fault injection for Windows readers that deny atomic JSON replacement."""
import ctypes
import json
import os
from pathlib import Path
import sys
import tempfile
import threading
import time
import unittest

sys.path.insert(0,str(Path(__file__).parents[1]/'tools'))
from prepare_ar_dataset import save


class StatusTest(unittest.TestCase):
    @unittest.skipUnless(os.name=='nt','Windows sharing semantics')
    def test_reader_lock_retries_without_truncating_visible_status(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'progress.json';save(path,dict(step=1))
            kernel=ctypes.WinDLL('kernel32',use_last_error=True)
            kernel.CreateFileW.argtypes=[ctypes.c_wchar_p,ctypes.c_uint32,ctypes.c_uint32,ctypes.c_void_p,ctypes.c_uint32,ctypes.c_uint32,ctypes.c_void_p]
            kernel.CreateFileW.restype=ctypes.c_void_p
            kernel.CloseHandle.argtypes=[ctypes.c_void_p]
            handle=kernel.CreateFileW(str(path),0x80000000,3,None,3,0,None) # Read/write sharing, no delete sharing.
            self.assertNotEqual(handle,ctypes.c_void_p(-1).value)
            errors=[]
            def writer():
                try:save(path,dict(step=2))
                except Exception as error:errors.append(error)
            thread=threading.Thread(target=writer);thread.start()
            try:
                time.sleep(.12)
                self.assertEqual(json.loads(path.read_text()),dict(step=1))
                self.assertTrue(thread.is_alive())
            finally:kernel.CloseHandle(handle)
            thread.join(3);self.assertFalse(thread.is_alive());self.assertFalse(errors)
            self.assertEqual(json.loads(path.read_text()),dict(step=2))


if __name__=='__main__':unittest.main()
