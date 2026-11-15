"""Charge each serial destination cohort, including Decode residence."""
import unittest
from .common import ROOT,json
from .policy import LayerPolicy


class DestinationPlanTests(unittest.TestCase):
    def setUp(self):
        self.cost=json.loads((ROOT/'configs/revised_cost_model_v3.json').read_text())
        self.policy=LayerPolicy(dict(D_admission_batch=8,decode_prediction='BATCH_LOOKUP'),self.cost)
    def test_exact_complete_and_remainder_batches(self):
        p=self.policy.destination_plan(8,16,16)
        self.assertEqual(p['batches'],[8,8])
        self.assertAlmostEqual(p['suffix_ms'],2*self.cost['D_suffix_ms']['8']['8'])
        self.assertAlmostEqual(p['intercohort_decode_ms'],15*self.policy.decode_bound(8))
        self.assertAlmostEqual(p['last_cohort_decode_ms'],15*self.policy.decode_bound(8))
        self.assertEqual(self.policy.destination_plan(4,10,64)['batches'],[8,2])
    def test_native_still_pays_intercohort_decode_wait(self):
        p=self.policy.destination_plan(0,16,64)
        self.assertEqual(p['suffix_ms'],0.)
        self.assertGreater(p['intercohort_decode_ms'],0.)
    def test_one_batch_does_not_pay_extra_decode_wait(self):
        p=self.policy.destination_plan(12,4,16)
        self.assertEqual(p['intercohort_decode_ms'],0.)


if __name__=='__main__':unittest.main()
