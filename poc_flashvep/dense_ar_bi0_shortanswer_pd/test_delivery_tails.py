import unittest
from .delivery_tails import intervals


class DeliveryTailTests(unittest.TestCase):
    def test_exact_clock_identity(self):
        row=dict(prompt_id='q',output_tokens=2,token_events=[
            dict(count=1,time=10.001,worker_time=10.),dict(count=2,time=10.050,worker_time=10.008)])
        values,missing=intervals(row)
        self.assertEqual(missing,0)
        self.assertAlmostEqual(values[0]['client_ITL_ms'],49.)
        self.assertAlmostEqual(values[0]['worker_emission_interval_ms'],8.)
        self.assertAlmostEqual(values[0]['delivery_lag_increment_ms'],41.)

    def test_multi_token_worker_clock_is_not_invented(self):
        row=dict(prompt_id='q',output_tokens=4,token_events=[
            dict(count=2,time=10.,worker_time=10.),dict(count=4,time=10.010,worker_time=10.009)])
        values,missing=intervals(row)
        self.assertEqual(values,[]);self.assertEqual(missing,3)
