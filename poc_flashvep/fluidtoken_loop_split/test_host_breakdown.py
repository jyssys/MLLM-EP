import unittest
from .host_breakdown import partition, request_key


class HostBreakdown(unittest.TestCase):
    def test_native_request_identifier(self):
        self.assertEqual(request_key(dict(run=0, sample_id=147)), 'r0_q147')

    def test_conserves_request_interval(self):
        row = dict(arrival=10., first_token=16., finish=20.)
        metric = dict(created_monotonic=10., timings=dict(
            admitted=1., prefill_started=2., prefill_compute_completed=4., d_activated=5.))
        values, total = partition(row, metric)
        self.assertEqual(sum(values.values()), total)
        self.assertEqual(values['P_observed_service'], 2.)

    def test_rejects_out_of_order_events(self):
        row = dict(arrival=10., first_token=16., finish=20.)
        metric = dict(created_monotonic=10., timings=dict(
            admitted=1., prefill_started=2., prefill_compute_completed=7., d_activated=8.))
        with self.assertRaises(ValueError):
            partition(row, metric)

    def test_activation_receipt_after_first_output_is_not_a_boundary(self):
        row = dict(arrival=10., first_token=16., finish=20.)
        metric = dict(created_monotonic=10., timings=dict(
            admitted=1., prefill_started=2., prefill_compute_completed=4., d_activated=6.1))
        values, total = partition(row, metric)
        self.assertEqual(values['P_completion_to_first_token'], 2.)
        self.assertEqual(sum(values.values()), total)


if __name__ == '__main__':
    unittest.main()
