"""Exercise the real reused orphan killer against isolated CPU child jobs."""
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from poc_dllm_prefix_canvas_cp.scripts.run_guarded import finish_this_run


class CleanupScope(unittest.TestCase):
    def test_cleanup_never_kills_another_output(self):
        import poc_multibd_moe_ep.scripts.run_guarded as cleanup
        import poc_native_lp_ep4.scripts.gpu_session as old
        with tempfile.TemporaryDirectory(prefix='prefix-canvas-cleanup-test-') as directory:
            root=Path(directory);out=root/'A';out.mkdir()
            children=[]
            try:
                for name in ['A','B']:
                    children.append(subprocess.Popen([sys.executable,'-c','import time; time.sleep(30)'],
                        env=dict(os.environ,POC_MULTIBD_OUTPUT=str(root/name)+'/worker')))
                compute=lambda:{p.pid for p in children if p.poll() is None}
                with patch.object(cleanup,'ROOT',root),patch.object(cleanup,'MASK','4,5,6,7'),patch.object(old,'compute_pids',compute):
                    finish_this_run(cleanup,out)
                self.assertIsNotNone(children[0].poll(),'Own orphan should terminate')
                self.assertIsNone(children[1].poll(),'Another output must remain alive')
            finally:
                for proc in children:
                    if proc.poll() is None:proc.terminate()
                    proc.wait(timeout=5)


if __name__=='__main__':unittest.main()
