import json
import tempfile
import unittest
from pathlib import Path
from .dynamic_campaign import cached_calibration


class CalibrationCacheTests(unittest.TestCase):
    def test_completed_immutable_predictor_is_not_a_new_measurement(self):
        with tempfile.TemporaryDirectory() as temporary:
            folder = Path(temporary)
            cfg = dict(name='same', depth=32)
            for name, value in [('arc_contract.json', cfg),
                    ('dispatcher_result.json', dict(complete=True)),
                    ('isolated_split_cost.json', dict(passed=True,
                        output=dict(reference='native'), cost=dict(p_service_s=.5)))]:
                (folder / name).write_text(json.dumps(value))
            result = cached_calibration(folder, cfg, 'native')
            self.assertIn('No new measurement', result['reuse_scope'])
            self.assertIsNone(cached_calibration(folder, cfg, 'different'))
            with self.assertRaisesRegex(AssertionError, 'contract changed'):
                cached_calibration(folder, dict(cfg, depth=4), 'native')
            (folder / 'dispatcher_result.json').write_text(json.dumps(dict(complete=False)))
            self.assertIsNone(cached_calibration(folder, cfg, 'native'))


if __name__ == '__main__':
    unittest.main()
