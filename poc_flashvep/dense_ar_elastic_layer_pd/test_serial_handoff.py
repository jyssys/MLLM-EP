"""Check the predictor against the actual no-active-cohort ingress contract."""
import unittest
from .common import *
from .policy import LayerPolicy,Observation


class SerialHandoffTest(unittest.TestCase):
    def setUp(self):
        self.cfg=dict(policy='REVISED',length=8000,cap=64,P_group_size=4,
            decode_prediction='BATCH_LOOKUP',revised_revision=2,maximum_predicted_D_wait_ms=400.)
        self.cost=json.loads((ROOT/'configs/revised_cost_model_v1.json').read_text())
    def test_read_cannot_overlap_an_unadmitted_decode_cohort(self):
        state=Observation(100.,4,16,101.,0,20000,8,100.034,admitted_D_requests=8)
        result=LayerPolicy(self.cfg,self.cost).choose(state)
        row=next(c for c in result.candidates if c['layers']==8)
        expected=max(row['predicted_P_cut_ready'],row['D_available_upper'])+row['predicted_handoff_ms']/1000
        self.assertAlmostEqual(row['predicted_D_start'],expected,places=9)


if __name__=='__main__':unittest.main()
