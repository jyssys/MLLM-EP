import unittest
from .serving_audit import paired


class PairedStatisticsTests(unittest.TestCase):
    def rows(self,reference,candidate):
        return [dict(name=f'recovery_ouro_{arm}_seed{i}',complete=True,model='ouro',arrival_seed=i,
            SLO_goodput=value) for i in range(3) for arm,value in (('native',reference),('loop',candidate))]

    def test_zero_candidate_is_not_missing(self):
        result=next(r for r in paired(self.rows(1.,0.)) if r['reference']=='native')
        self.assertEqual(result['paired_starts'],3)
        self.assertEqual(result['geometric_ratio'],0.)
        self.assertEqual(result['bootstrap_95_CI'],[0.,0.])
        self.assertEqual(result['paired_mean_difference'],-1.)

    def test_zero_reference_is_explicit(self):
        result=next(r for r in paired(self.rows(0.,1.)) if r['reference']=='native')
        self.assertEqual(result['paired_starts'],3)
        self.assertIsNone(result['geometric_ratio'])
        self.assertEqual(result['ratio_status'],'UNDEFINED_ZERO_REFERENCE')
        self.assertEqual(result['mean_difference_bootstrap_95_CI'],[1.,1.])
