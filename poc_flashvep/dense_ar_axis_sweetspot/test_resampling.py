import unittest
from .resampling import mean_interval,ratio_interval,empirical_quantile


class BootstrapTests(unittest.TestCase):
    def test_three_start_extreme_draw_mass_is_not_interpolated_away(self):
        values=[.9197886734342793,.93284887555283,1.0000237259695057]
        result=mean_interval(values)
        self.assertEqual(result,[min(values),max(values)])
        self.assertGreaterEqual(result[1],1.)

    def test_paired_ratios_keep_independent_start_units(self):
        result=ratio_interval([1,2,4],[1,2.1,4.4])
        self.assertAlmostEqual(result['ratio'],7.5/7)
        self.assertAlmostEqual(result['ratio_CI95'][0],1.)
        self.assertAlmostEqual(result['ratio_CI95'][1],1.1)

    def test_zero_denominator_draw_is_not_silently_removed(self):
        result=ratio_interval([0,1,2],[0,1,3])
        self.assertIsNone(result['ratio_CI95'])
        self.assertEqual(result['ratio_CI_status'],'UNDEFINED_ZERO_DENOMINATOR_DRAWS')

    def test_single_start_has_no_CI(self):
        self.assertIsNone(mean_interval([1.]))
        self.assertIsNone(ratio_interval([1.],[2.])['ratio_CI95'])


if __name__=='__main__':
    unittest.main()
