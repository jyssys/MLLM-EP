"""Sampling may remove records, but must never remove real engine calls."""
from types import SimpleNamespace
import unittest
from unittest.mock import patch


class ObserverSamplingTest(unittest.TestCase):
    def test_every_real_schedule_and_execute_still_runs(self):
        from vllm.v1.core.sched.scheduler import Scheduler
        from .worker_observer import Collector
        called = []
        output = SimpleNamespace(num_scheduled_tokens={'r': 128}, total_num_scheduled_tokens=128,
                                 scheduled_encoder_inputs={})
        request = SimpleNamespace(status='WAITING', num_computed_tokens=1, num_prompt_tokens=512)
        scheduler = SimpleNamespace(waiting=[request], running=[], requests={'r': request})
        def original_schedule(*args, **kwargs):
            called.append('schedule')
            return output
        class Runner:
            def execute_model(self, *args, **kwargs):
                called.append('execute')
                return 'original-output'
        collector = Collector.__new__(Collector)
        collector.runner = Runner()
        collector.rows, collector.calls = [], []
        collector.state_interval_ns = 50_000_000
        collector.last_schedule_sample_ns = collector.last_execute_sample_ns = 0
        clock = {'ns': 100_000_000}
        with patch.object(Scheduler, 'schedule', original_schedule), \
                patch('poc_flashvep.mllm_epd_prefill_heavy_token_split.worker_observer.time.monotonic_ns',
                      side_effect=lambda: clock['ns']):
            collector.install_state_hooks()
            for timestamp in [100_000_000, 110_000_000, 160_000_000]:
                clock['ns'] = timestamp
                self.assertIs(Scheduler.schedule(scheduler), output)
                self.assertEqual(collector.runner.execute_model(output), 'original-output')
        self.assertEqual(called.count('schedule'), 3)
        self.assertEqual(called.count('execute'), 3)
        self.assertEqual(len(collector.rows), 2)
        self.assertEqual(len(collector.calls), 2)


if __name__ == '__main__':
    unittest.main()
