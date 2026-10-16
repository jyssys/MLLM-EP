"""Do not classify another process session as an owned GPU job."""
import os
import signal
import subprocess
import sys
import tempfile
import unittest
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock,patch
from poc_ouro_reasoning_fidelity_multigpu import guard_cleanup
from poc_ouro_reasoning_fidelity_multigpu.guard_cleanup import belongs,registered_launch


class GuardScopeTests(unittest.TestCase):
    def test_exact_registered_command_and_environment(self):
        groups=[]
        original=Mock(return_value=SimpleNamespace(pid=12345))
        launch=registered_launch(original,['model-probe'],Path('/scoped-output'),groups)
        launch(['other-probe'],env={'LOOPED_KV_OUTPUT':'/scoped-output'},start_new_session=True)
        launch(['model-probe'],env={'LOOPED_KV_OUTPUT':'/other-output'},start_new_session=True)
        self.assertEqual(groups,[])
        launch(['model-probe'],env={'LOOPED_KV_OUTPUT':'/scoped-output'},start_new_session=True)
        self.assertEqual(groups,[12345])

    def test_only_the_created_session_matches(self):
        process=subprocess.Popen([sys.executable,'-c','import time; time.sleep(30)'],start_new_session=True)
        try:
            self.assertTrue(belongs(process.pid,process.pid))
            self.assertFalse(belongs(os.getpid(),process.pid))
            self.assertEqual(os.getsid(process.pid),process.pid)
            self.assertEqual(os.getpgid(process.pid),process.pid)
        finally:
            os.killpg(process.pid,signal.SIGTERM)
            process.wait(timeout=5)

    def test_normal_cleanup_does_not_duplicate_a_pending_restore(self):
        # The original guard can already have restored a CPU-loading worker.
        # That worker does not have a visible GPU context yet.
        with tempfile.TemporaryDirectory() as temporary:
            with patch.object(guard_cleanup,'ASSETS',Path(temporary)), \
                 patch.dict(os.environ,{'PATH':'/test/bin'},clear=True), \
                 patch.object(guard_cleanup.subprocess,'Popen',return_value=SimpleNamespace(pid=12345)) as launch, \
                 patch.object(guard_cleanup,'write') as write:
                guard_cleanup.finish([], [4,5],Path(temporary)/'probe',lambda gpus:[])
            self.assertEqual(launch.call_count,0,'A normal restore must not create another launcher.')
            self.assertEqual(write.call_args.args[1]['scoped_idle_burner_restore'],[])

    def test_reclaimed_context_restore_uses_only_requested_mask(self):
        active=[True]
        def contexts(gpus):
            return [12345] if active[0] else []
        def stop(group,signum):
            active[0]=False
        with tempfile.TemporaryDirectory() as temporary:
            with patch.object(guard_cleanup,'ASSETS',Path(temporary)), \
                 patch.dict(os.environ,{'PATH':'/test/bin'},clear=True), \
                 patch.object(guard_cleanup,'belongs',return_value=True), \
                 patch.object(guard_cleanup.os,'killpg',side_effect=stop), \
                 patch.object(guard_cleanup.subprocess,'Popen',return_value=SimpleNamespace(pid=12345)) as launch, \
                 patch.object(guard_cleanup,'write') as write:
                guard_cleanup.finish([12345],[4,5],Path(temporary)/'probe',contexts)
            self.assertEqual(launch.call_args.kwargs['env']['CUDA_VISIBLE_DEVICES'],'4,5')
            self.assertEqual(launch.call_args.kwargs['env']['VLLM_UTILIZE_GPUS'],'4,5')
            self.assertEqual(write.call_args.args[1]['scoped_idle_burner_restore'],[4,5])

    def test_late_context_release_without_pending_restore_gets_burner(self):
        with tempfile.TemporaryDirectory() as temporary:
            output=Path(temporary)/'probe'
            output.mkdir()
            (output/'guard_status.json').write_text(json.dumps({'restored_burner_workers':[]}))
            with patch.object(guard_cleanup,'ASSETS',Path(temporary)), \
                 patch.dict(os.environ,{'PATH':'/test/bin'},clear=True), \
                 patch.object(guard_cleanup.subprocess,'Popen',return_value=SimpleNamespace(pid=12345)) as launch, \
                 patch.object(guard_cleanup,'write'):
                guard_cleanup.finish([], [4],output,lambda gpus:[])
            self.assertEqual(launch.call_count,1)

    def test_cpu_loading_restored_worker_prevents_duplicate(self):
        # A CPU-only stand-in has the same argv markers as burner bootstrap.
        process=subprocess.Popen([sys.executable,'-c','import time; time.sleep(30)',
            '/home/esjung/vllm-ep/utilize.py','--worker','--gpu','4'],start_new_session=True)
        try:
            with tempfile.TemporaryDirectory() as temporary:
                output=Path(temporary)/'probe'
                output.mkdir()
                (output/'guard_status.json').write_text(json.dumps({'restored_burner_workers':[
                    {'physical_gpu':4,'pid':process.pid}]}))
                with patch.object(guard_cleanup,'ASSETS',Path(temporary)), \
                     patch.dict(os.environ,{'PATH':'/test/bin'},clear=True), \
                     patch.object(guard_cleanup.subprocess,'Popen',return_value=SimpleNamespace(pid=12345)) as launch, \
                     patch.object(guard_cleanup,'write'):
                    guard_cleanup.finish([], [4],output,lambda gpus:[])
                self.assertEqual(launch.call_count,0)
        finally:
            os.killpg(process.pid,signal.SIGTERM)
            process.wait(timeout=5)

    def test_unrelated_context_is_not_signalled(self):
        with patch.object(guard_cleanup,'belongs',return_value=False), \
             patch.dict(os.environ,{'PATH':'/test/bin'},clear=True), \
             patch.object(guard_cleanup.time,'monotonic',side_effect=[0.,31.]), \
             patch.object(guard_cleanup.os,'killpg') as signal_group, \
             patch.object(guard_cleanup.subprocess,'Popen') as launch, \
             patch.object(guard_cleanup,'write'):
            guard_cleanup.finish([12345],[4],Path('/unused-test-output'),lambda gpus:[987654321])
        signal_group.assert_not_called()
        launch.assert_not_called()


if __name__=='__main__':
    unittest.main()
