import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from . import refresh_analysis, transport_timings


class ServingWorkerScopeTests(unittest.TestCase):
    def fixture(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        root = Path(temporary.name)
        for child in ('raw/cell', 'configs', 'state'):
            (root / child).mkdir(parents=True)
        cfg = dict(name='cell', model='ouro', worker='rlt_worker', load='heavy',
            count=1, observe_service_timing=True, state_probe=False, verify_transfer=False)
        (root / 'configs/cell.json').write_text(json.dumps(cfg))
        (root / 'raw/cell/dispatcher_result.json').write_text(json.dumps(dict(complete=True)))
        return root

    def test_refresh_does_not_discard_the_real_serving_worker(self):
        from . import metrics, work_accounting, lifetime_gate
        from ..looped_pd_atomic_prefill import evidence
        root = self.fixture()
        with patch.object(refresh_analysis, 'ROOT', root), patch.object(refresh_analysis, 'initialize'), \
                patch.object(metrics, 'summarize', return_value={}), \
                patch.object(work_accounting, 'summarize', return_value={}), \
                patch.object(lifetime_gate, 'check', return_value=dict(passed=True)), \
                patch.object(evidence, 'full_work', return_value=dict(passed=True)), \
                patch.object(evidence, 'compare', return_value=dict(passed=True)):
            refresh_analysis.main()
        rows = json.loads((root / 'state/refreshed_analysis.json').read_text())['results']
        self.assertEqual(len(rows), 1)
        self.assertTrue(rows[0]['passed'])

    def test_transport_analysis_does_not_discard_the_real_serving_worker(self):
        root = self.fixture()
        (root / 'raw/cell/lifetime_events.jsonl').write_text('')
        (root / 'raw/cell/closed_stats.json').write_text(json.dumps(dict(metrics=[
            dict(request_id='r0_q0', transfer_id='x')])))
        with patch.object(transport_timings, 'ROOT', root), patch.object(transport_timings, 'initialize'):
            transport_timings.main()
        rows = json.loads((root / 'state/host_transport_lifetime_timings.json').read_text())['results']
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]['offered'], 1)

    def test_heavy_calibration_and_nonserving_workers_remain_excluded(self):
        for override in (dict(state_probe=True), dict(verify_transfer=True),
                         dict(calibration_serial=True), dict(worker='collector_selftest')):
            with self.subTest(override=override):
                root = self.fixture()
                path = root / 'configs/cell.json'
                path.write_text(json.dumps(dict(json.loads(path.read_text()), **override)))
                with patch.object(refresh_analysis, 'ROOT', root), patch.object(refresh_analysis, 'initialize'):
                    refresh_analysis.main()
                rows = json.loads((root / 'state/refreshed_analysis.json').read_text())['results']
                self.assertEqual(rows, [])


if __name__ == '__main__':
    unittest.main()
