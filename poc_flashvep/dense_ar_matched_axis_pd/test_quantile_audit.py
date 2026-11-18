"""Test the observed boundary without changing the historical statistic."""
import itertools
import unittest
from .quantile_audit import inverse_cdf,cdf_ratio_interval
from ..dense_ar_layer_token_pd.statistics_report import draws,ratio_interval


class QuantileAuditTests(unittest.TestCase):
    def test_revised_batch_protocol_is_never_pooled_with_original(self):
        from .statistics import analysis_stage
        for stage in ('batch1_discovery', 'batch1_confirmation', 'cohort2_discovery',
                      'cohort2_confirmation', 'confirmation', 'admission_control'):
            self.assertEqual(analysis_stage(stage), stage)
        self.assertEqual(analysis_stage('motivation_odd'), 'discovery')

    def test_exact_draws_are_independent_start_indices(self):
        values=list(draws(3))
        self.assertEqual(len(values),27)
        self.assertEqual(set(values),set(itertools.product(range(3),repeat=3)))

    def test_observed_linear_boundary_is_not_a_robust_win(self):
        x=[10637.544236611575,11355.934631079435,9237.996048294008]
        y=[10149.141512811184,10956.577030941844,9339.246657211334]
        self.assertLess(ratio_interval(x,y)['ratio_CI95'][1],1)
        self.assertGreater(cdf_ratio_interval(x,y)[1],1)
        self.assertEqual(cdf_ratio_interval(x,y),[min(b/a for a,b in zip(x,y)),max(b/a for a,b in zip(x,y))])

    def test_inverse_cdf_replication_invariance(self):
        values=list(range(27))
        for q in (0,.025,.5,.975,1):
            self.assertEqual(inverse_cdf(values,q),inverse_cdf(values*1000,q))

    def test_zero_and_single_start_have_no_ratio_interval(self):
        self.assertIsNone(cdf_ratio_interval([1],[1]))
        self.assertIsNone(cdf_ratio_interval([0,1],[1,1]))


if __name__=='__main__':
    unittest.main()
