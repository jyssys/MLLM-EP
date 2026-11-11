import unittest
from .paired_host_attribution import decode_bins, durations


class PairedHostAttributionTest(unittest.TestCase):
    def test_clock_origin_and_signed_receipts(self):
        t = dict(admitted=.1, prefill_started=.2, prefill_compute_completed=.7,
                 prefill_submitted=.6, d_received=.8)
        parts = durations(10., 11.1, 10.1, t)
        self.assertAlmostEqual(sum(parts.values()), 1100.)
        self.assertAlmostEqual(parts['arrival_to_admit'], 200.)
        self.assertAlmostEqual(parts['P_release_to_wire_commit_receipt'], -100.)
        del t['d_received']
        self.assertIsNone(durations(10., 11.1, 10.1, t))

    def test_live_decode_is_not_submission_batch(self):
        rows = [
            dict(first_token=0., finish=3., token_events=[
                dict(first=False, previous_time=0., time=1.),
                dict(first=False, previous_time=1., time=3.)]),
            dict(first_token=1., finish=2., token_events=[
                dict(first=False, previous_time=1., time=2.)]),
        ]
        result = decode_bins(rows)
        self.assertEqual(result[1]['n'], 1)
        self.assertEqual(result[2]['n'], 2)
        self.assertEqual(result[2]['maximum'], 2000.)


if __name__ == '__main__':
    unittest.main()
