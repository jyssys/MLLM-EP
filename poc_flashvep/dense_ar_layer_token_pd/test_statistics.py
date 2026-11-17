"""Test paired-start ratios and undefined denominators without GPU work."""
import unittest
from .statistics_report import draws,ratio_interval


class StartStatisticsTests(unittest.TestCase):
    def test_three_start_resampling_has_twenty_seven_draws(self):
        self.assertEqual(len(list(draws(3))),27)

    def test_constant_paired_ratio(self):
        value=ratio_interval([1.,2.,3.],[2.,4.,6.])
        self.assertEqual(value['ratio'],2.)
        self.assertEqual(value['ratio_CI95'],[2.,2.])
        self.assertEqual(value['starts'],3)

    def test_ratio_uses_start_means(self):
        value=ratio_interval([1.,9.,10.],[2.,9.,10.])
        self.assertAlmostEqual(value['ratio'],1.05)
        self.assertNotAlmostEqual(value['ratio'],(2+1+1)/3)

    def test_single_start_has_no_resampling_interval(self):
        value=ratio_interval([1.],[2.])
        self.assertIsNone(value['ratio_CI95'])
        self.assertEqual(value['ratio_CI_status'],'NOT_ESTIMATED_SINGLE_START')

    def test_zero_reference_is_not_missing_measurement(self):
        value=ratio_interval([0.,0.,0.],[1.,2.,3.])
        self.assertIsNone(value['ratio'])
        self.assertEqual(value['ratio_status'],'UNDEFINED_ZERO_REFERENCE_MEAN')
        self.assertIsNone(value['ratio_CI95'])
        self.assertEqual(value['zero_denominator_draws'],27)
        self.assertIsNotNone(value['difference_CI95'])


if __name__=='__main__':
    unittest.main()
