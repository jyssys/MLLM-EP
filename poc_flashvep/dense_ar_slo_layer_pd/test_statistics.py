"""Check independent-start summaries without treating tokens as replicates."""
import unittest
from .statistics import paired_ratio
from .final_analysis import mean_CI


class StartStatisticsTests(unittest.TestCase):
    def test_three_pairs_have_twenty_seven_empirical_draws(self):
        value=paired_ratio([8.,9.,10.],[10.,10.,10.])
        self.assertEqual(value['independent_start_pairs'],3)
        self.assertEqual(value['bootstrap_draws'],27)
        self.assertAlmostEqual(value['ratio_of_means'],.9)
        self.assertGreaterEqual(value['paired_bootstrap_95_percentile_CI'][0],.8)
        self.assertLessEqual(value['paired_bootstrap_95_percentile_CI'][1],1.)

    def test_constant_start_metric_has_constant_empirical_interval(self):
        self.assertEqual(mean_CI([5.,5.,5.]),dict(samples=3,mean=5.,CI95=[5.,5.]))

    def test_missing_metric_stays_missing(self):
        self.assertEqual(mean_CI([]),dict(samples=0,mean=None,CI95=None))


if __name__=='__main__':unittest.main()
