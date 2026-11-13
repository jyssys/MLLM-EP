"""Check smaller-cut fallback with the frozen measured cost fixture."""
import unittest
from .common import *
from .policy import LayerPolicy,Observation


class RevisedPolicyTest(unittest.TestCase):
    def setUp(self):
        self.cfg=dict(policy='REVISED',length=8000,cap=64,P_group_size=4,decode_prediction='BATCH_LOOKUP')
        self.cost=json.loads((ROOT/'configs/revised_cost_model_v1.json').read_text())
    def choose(self,drain,**changes):
        state=dict(now=100.,group_size=4,P_backlog=16,D_available=100.+drain,pending_requests=0,
            KV_free_blocks=20000,active_decode=8,earliest_decode_deadline=100.034,admitted_D_requests=8)
        state.update(changes);return LayerPolicy(self.cfg,self.cost).choose(Observation(**state))
    def test_all_cut_sizes_can_follow_state(self):
        for drain,expected in ((.5,12),(.9,8),(1.1,4),(1.2,2),(1.4,0)):
            with self.subTest(drain=drain):self.assertEqual(self.choose(drain).layers,expected)
    def test_smaller_cut_is_not_forced(self):
        result=self.choose(1.1)
        self.assertEqual(result.reason,'ONLINE_BOUNDED_QUEUE_AWARE')
        self.assertFalse(next(c for c in result.candidates if c['layers']==12)['accepted'])
        self.assertTrue(next(c for c in result.candidates if c['layers']==4)['accepted'])
    def test_stale_state(self):self.assertEqual(self.choose(0,state_age_ms=251).reason,'STALE_D_STATE')
    def test_KV_credit(self):self.assertEqual(self.choose(0,KV_free_blocks=0).reason,'KV_CREDIT_BOUND')
    def test_no_observed_queue(self):self.assertEqual(self.choose(0,P_backlog=0).layers,0)
    def test_candidate_order_does_not_force_small_cuts(self):
        self.cfg['layer_candidates']=[12,8,4,2]
        self.assertEqual(self.choose(1.1).layers,4)
    def test_independent_budgets_and_uncertainty(self):
        result=self.choose(.9)
        for row in result.candidates:
            self.assertIn('admission_budget_ms',row);self.assertIn('execution_budget_ms',row)
            self.assertEqual(row['safety_margin_ms'],110.)
            self.assertGreater(row['predicted_D_finish'],row['predicted_D_finish_mean'])
            if row['accepted']:
                self.assertLessEqual(row['D_available_upper'],row['source_earliest_ready'])
    def test_queue_benefit_ablation(self):
        self.cfg['queue_benefit_weight']=0.
        result=self.choose(1.1)
        self.assertEqual(result.layers,0)
    def test_zero_request_budget(self):
        self.cfg['request_budget_fraction']=0.
        self.assertEqual(self.choose(.5).layers,0)
    def test_matched_static_assignment_is_explicit(self):
        self.cfg['static_layers_by_request']=[0,0,2,2,4,4,8,8,12,12]
        policy=LayerPolicy(self.cfg,self.cost)
        self.assertEqual([policy.static_request_layers(i) for i in range(10)],self.cfg['static_layers_by_request'])
    def test_cut_specific_margin_changes_upper_not_mean(self):
        before=self.choose(.5)
        self.cfg['safety_margin_by_layers_ms']={'2':110.,'4':80.,'8':60.,'12':90.}
        after=self.choose(.5)
        for a,b in zip(before.candidates,after.candidates):
            self.assertEqual(a['predicted_D_finish_mean'],b['predicted_D_finish_mean'])
            self.assertEqual(b['safety_margin_ms'],self.cfg['safety_margin_by_layers_ms'][str(b['layers'])])
    def test_continuous_native_counterfactual_does_not_wait_for_drain(self):
        self.cfg['native_continuous_D_ingress']=True
        first=self.choose(.5);second=self.choose(1.1)
        self.assertEqual(first.candidates[0]['predicted_Native_finish'],second.candidates[0]['predicted_Native_finish'])
    def test_control_reply_wait_does_not_relax_any_safety_gate(self):
        before=self.choose(.5)
        self.cfg['exclude_negotiation_from_D_occupancy']=True
        after=self.choose(.5)
        for a,b in zip(before.candidates,after.candidates):
            self.assertEqual(a['accepted'],b['accepted'])
            self.assertEqual(a['predicted_D_finish'],b['predicted_D_finish'])
            self.assertAlmostEqual(a['predicted_D_incremental_cost_ms']-b['predicted_D_incremental_cost_ms'],a['negotiation_bound_ms'])
    def test_layer_budget_also_evaluates_smaller_cuts(self):
        self.cfg['layer_budget_per_request']=4.
        self.assertEqual(self.choose(.5).layers,4)
    def test_cumulative_request_budget(self):
        self.cfg['request_budget_fraction']=.5
        policy=LayerPolicy(self.cfg,self.cost)
        state=Observation(100.,4,16,100.5,0,20000,8,100.034,admitted_D_requests=8)
        selected=0
        for step in range(1,9):
            result=policy.choose(state)
            selected+=4*bool(result.layers)
            self.assertLessEqual(selected,int(step*4*.5))


if __name__=='__main__':unittest.main()
