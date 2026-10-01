import os
from pathlib import Path
import subprocess
import sys
import unittest


class HookFailureTest(unittest.TestCase):
    def test_startup_hook_cannot_silently_fallback(self):
        source=Path('poc_native_lp_ep4/hooks/sitecustomize.py').read_text()
        code="""
import os, sys, types
os.environ['NATIVE_LP_ENABLE']='1'
module=types.ModuleType('poc_native_lp_ep4.native')
def install(): raise RuntimeError('deliberate integration failure')
module.install=install
sys.modules['poc_native_lp_ep4.native']=module
"""+source
        env=os.environ.copy()
        env.pop('PYTHONPATH',None)
        env.pop('NATIVE_LP_ENABLE',None)
        result=subprocess.run([sys.executable,'-S','-c',code],env=env,
                              capture_output=True,text=True,timeout=10)
        self.assertEqual(result.returncode,73)
        self.assertIn('deliberate integration failure',result.stderr)

    def test_native_header_uses_mandatory_dp_sync_not_extra_collective(self):
        # Static guard supplements (does not replace) physical wave-transition
        # regression runs. An extra conditional model collective was unsafe.
        source=Path('poc_native_lp_ep4/native.py').read_text()
        self.assertIn('dp_utils._run_ar=coordinated_header',source)
        self.assertNotIn('dist.all_gather(',source)


if __name__=='__main__':unittest.main()
