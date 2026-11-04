"""Fail closed when an online policy lacks matching measured calibration."""
import copy
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from . import dynamic_adapter as adapter


class CalibrationContract(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.patch = patch.object(adapter, 'ROOT', self.root)
        self.patch.start()
        self.addCleanup(self.patch.stop)
        (self.root / 'configs').mkdir()
        (self.root / 'configs/frozen_SLO.json').write_text(json.dumps({
            'huginn_L2048': dict(TTFT_ms=2000., TPOT_ms=200., ITL_ms=220.)}))
        evidence = self.root / 'calibration.json'
        evidence.write_text('{"scope":"UNIT_TEST_FIXTURE_NOT_GPU_MEASUREMENT"}')
        self.cfg = dict(model='huginn', length=2048, depth=32, axis='loop',
                        dynamic_cost_table=str(self.root / 'table.json'))
        self.table = dict(qualified=True, axis='loop',
            execution_signature={key: self.cfg.get(key) for key in adapter.SIGNATURE},
            TTFT_limit_s=2., TPOT_margin_s=.18, ITL_limit_s=.22,
            safe_decode_batch=4, baseline_token_interval_s=.1,
            calibration_evidence={str(evidence): adapter.digest(evidence)},
            costs=[dict(key=31, p_service_s=.8, d_continuation_s=.02,
                        transfer_exposed_s=.01, d_extra_blocks=0),
                   dict(key=32, p_service_s=1., d_continuation_s=0.,
                        transfer_exposed_s=.01, d_extra_blocks=0)])

    def load(self, table=None):
        path = Path(self.cfg['dynamic_cost_table'])
        path.write_text(json.dumps(self.table if table is None else table))
        self.cfg['dynamic_cost_table_SHA256'] = adapter.digest(path)
        return adapter.load_table(self.cfg)

    def test_matching_contract(self):
        _, costs, full = self.load()
        self.assertEqual([row.key for row in costs], [31, 32])
        self.assertEqual(full.key, 32)

    def test_reject_invalid_contracts(self):
        mutations = [
            lambda t: t.update(qualified=False),
            lambda t: t.update(ITL_limit_s=.3),
            lambda t: t.update(baseline_token_interval_s=float('inf')),
            lambda t: t.update(calibration_evidence={}),
            lambda t: t['execution_signature'].update(graph_rows=8),
            lambda t: t['costs'][0].update(key=0),
            lambda t: t['costs'][0].update(d_continuation_s=float('nan')),
            lambda t: t['costs'][0].update(d_extra_blocks=.5),
            lambda t: t['costs'][1].update(d_continuation_s=.1),
        ]
        for mutate in mutations:
            with self.subTest(mutation=mutate):
                value = copy.deepcopy(self.table)
                mutate(value)
                with self.assertRaises(AssertionError):
                    self.load(value)

    def test_reject_changed_evidence(self):
        path = next(iter(self.table['calibration_evidence']))
        Path(path).write_text('{"changed":true}')
        with self.assertRaises(AssertionError):
            self.load()


if __name__ == '__main__':
    unittest.main()
