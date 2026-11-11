import unittest
from .scientific_gates import check_ratio


class RatioGateTests(unittest.TestCase):
    def test_TTFT_must_meet_target_and_exclude_one(self):
        row = dict(geometric_ratio=.94, paired_starts=3, bootstrap_95_CI=[.90,1.])
        self.assertEqual(check_ratio(row,.95,1.)['status'], 'FAIL')
        self.assertEqual(check_ratio(dict(row, bootstrap_95_CI=[.90,.98]),.95,1.)['status'], 'PASS')
        self.assertEqual(check_ratio(dict(row, geometric_ratio=.97),.95,1.)['status'], 'FAIL')

    def test_decode_noninferiority_uses_the_upper_interval(self):
        row = dict(geometric_ratio=1.02, paired_starts=3, bootstrap_95_CI=[.99,1.06])
        self.assertEqual(check_ratio(row,1.05,1.05)['status'], 'FAIL')
        self.assertEqual(check_ratio(dict(row, bootstrap_95_CI=[.99,1.05]),1.05,1.05)['status'], 'PASS')

    def test_missing_or_short_evidence_cannot_pass(self):
        self.assertEqual(check_ratio(None,.95,1.)['status'], 'NOT_MEASURED')
        self.assertEqual(check_ratio(dict(geometric_ratio=.85, paired_starts=2),.95,1.)['status'],
            'INSUFFICIENT_INDEPENDENT_STARTS')


if __name__ == '__main__':
    unittest.main()
