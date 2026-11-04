import unittest
from types import SimpleNamespace as NS
from .service_timing import device_time, calibrated_device_time, prompt_span


class ServiceTimingTests(unittest.TestCase):
    def test_device_clock_bounds(self):
        self.assertEqual(device_time(dict(lower=10., upper=10.0002, midpoint=10.0001), 500.),
                         dict(time=10.5001, lower=10.5, upper=10.5002))

    def test_prefill_and_decode_classification(self):
        request = NS(generated_token_ids=[])
        span = NS(chunk_start=None, token_count=2048, request=request)
        self.assertTrue(prompt_span(NS(operation=NS(value='prelude')), span))
        span.token_count = 1
        request.generated_token_ids = [1]
        self.assertFalse(prompt_span(NS(operation=NS(value='prelude')), span))
        span.chunk_start = 1984
        self.assertTrue(prompt_span(NS(operation=NS(value='recurrent')), span))

    def test_endpoints_correct_linear_clock_drift(self):
        start = dict(midpoint=10., lower=9.999, upper=10.001)
        end = dict(midpoint=12.002, lower=12.001, upper=12.003)
        value = calibrated_device_time(start, end, 2000., 1000.)
        self.assertAlmostEqual(value['time'], 11.001)
        self.assertAlmostEqual(value['lower'], 11.)
        self.assertAlmostEqual(value['upper'], 11.002)
        with self.assertRaises(ValueError):
            calibrated_device_time(start, end, 2000., 2001.)


if __name__ == '__main__':
    unittest.main()
