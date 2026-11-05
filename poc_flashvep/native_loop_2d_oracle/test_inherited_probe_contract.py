"""Reproduce the actual inherited D scheduler change without CUDA execution."""
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import patch
from .common import parent

sys.path.insert(0, str(parent.prior.RLT))


class ProbeContractTests(unittest.TestCase):
    def construct(self, enabled):
        from vllm_rlt.config import SchedulerConfig
        from vllm_rlt.engine import stage_engine
        from vllm_rlt.worker import model_runner
        from ..fluidtoken_loop_split import common, probe_quantum

        class FakeRunner:
            def submit(self, *args, **kwargs):
                return None

        class FakeEngine:
            def __init__(self, *, stage_config, scheduler_config):
                self.stage_config = stage_config
                self.scheduler = NS(wavefront_chunked_prefill=scheduler_config.wavefront_chunked_prefill,
                                    requests={})
                self.model_runner = FakeRunner()

            def step(self):
                return []

            def close(self):
                return None

            def _execution_ready(self, request):
                return True

        with tempfile.TemporaryDirectory(prefix='native_probe_contract_') as temporary:
            with patch.object(common, 'ROOT', Path(temporary)), \
                 patch.object(stage_engine, 'StageEngine', FakeEngine), \
                 patch.object(model_runner, 'ModelRunner', FakeRunner):
                probe_quantum.install(dict(name='fixture', D_local_probe_output_pacing=enabled))
                engine = FakeEngine(stage_config=NS(role='decode'),
                    scheduler_config=SchedulerConfig(mode='refill', wavefront_chunked_prefill=False))
                return engine.scheduler.wavefront_chunked_prefill

    def test_inherited_enabled_flag_reproduces_actual_wavefront_change(self):
        self.assertTrue(self.construct(True))

    def test_disabled_unused_probe_hook_keeps_requested_native_scheduler(self):
        self.assertFalse(self.construct(False))


if __name__ == '__main__':
    unittest.main()
