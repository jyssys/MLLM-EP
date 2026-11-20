import unittest
from .confirmation_analysis import paired_ratio


class ConfirmationStatisticsTests(unittest.TestCase):
    def test_paired_scale_is_not_destroyed_by_resampling(self):
        result=paired_ratio([2.,4.,8.],[1.,2.,4.])
        self.assertEqual(result['ratio'],2.)
        self.assertEqual(result['CI95'],[2.,2.])

    def test_one_start_has_no_invented_interval(self):
        self.assertIsNone(paired_ratio([1.],[1.])['CI95'])

    def test_zero_goodput_denominator_retains_difference(self):
        result=paired_ratio([1.,2.,3.],[0.,0.,0.])
        self.assertIsNone(result['ratio']);self.assertIsNone(result['CI95'])
        self.assertEqual(result['difference'],2.)
