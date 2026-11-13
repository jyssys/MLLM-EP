import unittest
from .statistics import paired_ratio,paired_difference


class PairedTests(unittest.TestCase):
    def test_constant_ratio(self):
        r=paired_ratio([2,4,6],[1,2,3])
        self.assertEqual(r['ratio_of_means'],2)
        self.assertEqual(r['paired_bootstrap_95_percentile_CI'],[2,2])
        self.assertEqual(r['bootstrap_draws'],27)
    def test_constant_difference(self):
        r=paired_difference([2,3,4],[1,2,3])
        self.assertEqual(r['mean_difference'],1)
        self.assertEqual(r['paired_bootstrap_95_percentile_CI'],[1,1])
    def test_pairing_matters(self):
        r=paired_ratio([1,8,3],[2,4,6])
        self.assertEqual(r['ratio_of_means'],1)
        self.assertLess(r['paired_bootstrap_95_percentile_CI'][0],1)
        self.assertGreater(r['paired_bootstrap_95_percentile_CI'][1],1)


if __name__=='__main__':unittest.main()
