import unittest
from .metrics import decompose, measured_device_rows


class MetricTests(unittest.TestCase):
    def test_request_identity(self):
        result = decompose(10., 12., dict(time=11.5, lower=11.499, upper=11.501))
        self.assertEqual(result['queue_ms'], 1500.)
        self.assertEqual(result['service_TTFT_ms'], 500.)
        self.assertEqual(result['TTFT_ms'], 2000.)
        self.assertAlmostEqual(result['start_clock_uncertainty_ms'], 1.)

    def test_invalid_order_is_not_clipped(self):
        for time in (9., 13.):
            with self.assertRaisesRegex(ValueError, 'timestamp order'):
                decompose(10., 12., dict(time=time, lower=time, upper=time))

    def test_startup_seed_reuse_is_not_a_measured_duplicate(self):
        warmup = dict(seed=7, request_id='startup_uuid', first_host_submit=9.)
        measured = dict(seed=7, request_id='measured_uuid', first_host_submit=11.)
        self.assertEqual(measured_device_rows([warmup, measured], {7}, 10.), {7: measured})

    def test_real_measured_duplicate_still_fails(self):
        first = dict(seed=7, request_id='first_uuid', first_host_submit=11.)
        second = dict(seed=7, request_id='second_uuid', first_host_submit=12.)
        with self.assertRaisesRegex(AssertionError, 'Duplicate measured'):
            measured_device_rows([first, second], {7}, 10.)


if __name__ == '__main__':
    unittest.main()
