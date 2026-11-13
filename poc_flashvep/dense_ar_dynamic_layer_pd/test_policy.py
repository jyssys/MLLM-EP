"""Test selection mechanics with synthetic costs, not claimed GPU results."""
import unittest
from .policy import LayerPolicy,Observation


def fixture():
    return dict(P_prefix_ms={'1':{'0':120,'2':100,'4':90,'8':70,'12':50}},
        D_suffix_ms={'1':{'2':10,'4':20,'8':40,'12':60}},
        handoff_ms={'1':{'0':30,'2':15,'4':15,'8':15,'12':15}},negotiation_idle_ms=0,negotiation_busy_ms=0)


class PolicyTests(unittest.TestCase):
    def config(self,**kw):return dict(policy='COST',length=8000,cap=16,safety_margin_ms=0,**kw)
    def test_slack_changes_the_selected_layer_count(self):
        selected=[]
        for available in (0.109,0.099,0.079,0.059):
            s=Observation(0,1,4,available,0,10000,1)
            selected.append(LayerPolicy(self.config(),fixture()).choose(s).layers)
        self.assertEqual(selected,[2,4,8,12])
    def test_absent_backlog_uses_native(self):
        p=LayerPolicy(self.config(),fixture());self.assertEqual(p.choose(Observation(0,1,0,0,0,10000,0)).layers,0)
    def test_stale_state_uses_native(self):
        p=LayerPolicy(self.config(),fixture());self.assertEqual(p.choose(Observation(0,1,3,0,0,10000,0,state_age_ms=300)).layers,0)
    def test_credit_failure_uses_native(self):
        p=LayerPolicy(self.config(),fixture());self.assertEqual(p.choose(Observation(0,1,3,0,0,0,0)).layers,0)
    def test_transfer_cost_can_reject_a_cut(self):
        c=fixture();c['handoff_ms']['1'].update({str(n):200 for n in (2,4,8,12)})
        p=LayerPolicy(self.config(),c);self.assertEqual(p.choose(Observation(0,1,3,0,0,10000,0)).layers,0)
    def test_static_fraction_is_distributed(self):
        p=LayerPolicy(dict(policy='STATIC',static_layers=8,migration_fraction=.25),fixture())
        got=[p.choose(Observation(0,1,3,0,0,10000,0)).layers for _ in range(8)]
        self.assertEqual(got,[0,0,0,8,0,0,0,8])
    def test_static_request_mask_has_exact_primary_counts(self):
        p=LayerPolicy(dict(policy='STATIC',static_layers=12,migration_fraction=.3125,static_request_block_size=4),fixture())
        self.assertEqual(sum(bool(p.static_request_layers(i)) for i in range(64)),20)
        self.assertEqual(sum(bool(p.static_request_layers(i)) for i in range(128)),40)
        self.assertTrue(all(p.static_request_layers(i)==p.static_request_layers(i//4*4) for i in range(128)))
    def test_forced_cuts_are_not_labeled_dynamic(self):
        p=LayerPolicy(dict(policy='FORCED_QUALIFICATION',forced_layers=[0,2,4,8,12]),fixture())
        self.assertEqual(p.choose(Observation(0,1,0,0,0,0,0)).reason,'FORCED_CORRECTNESS_NOT_DYNAMIC')
    def test_different_pending_cut_can_finish_before_handoff(self):
        p=LayerPolicy(self.config(),fixture())
        state=Observation(0,1,4,.059,2,10000,0,pending_cuts=(36,36))
        self.assertEqual(p.choose(state).layers,12)
    def test_pending_completion_still_blocks_unsafe_prediction(self):
        p=LayerPolicy(self.config(),fixture())
        state=Observation(0,1,4,.2,2,10000,1,pending_cuts=(36,36))
        self.assertEqual(p.choose(state).layers,0)
    def test_batch_lookup_uses_a_measured_upper_shape(self):
        costs=fixture();costs['decode_step_p95_ms']=30
        costs['decode_step_by_batch_p95_ms']={'1':24,'4':26,'16':30}
        p=LayerPolicy(self.config(decode_prediction='BATCH_LOOKUP',decode_prediction_margin_ms=2),costs)
        self.assertEqual(p.decode_bound(2),28)
        self.assertEqual(p.decode_bound(16),32)
        self.assertEqual(p.decode_bound(17),32)
    def test_global_bound_is_unchanged_by_default(self):
        costs=fixture();costs['decode_step_p95_ms']=30
        self.assertEqual(LayerPolicy(self.config(),costs).decode_bound(2),30)
    def test_rejection_log_keeps_all_failed_gates(self):
        costs=fixture();costs['handoff_ms']['1']['2']=35
        p=LayerPolicy(self.config(),costs)
        c=p.choose(Observation(0,1,4,.2,2,10000,1)).candidates[0]
        self.assertIn('INSUFFICIENT_IDLE_PREDICTION',c['rejection_reasons'])
        self.assertIn('NO_COMPLETION_BENEFIT',c['rejection_reasons'])


if __name__=='__main__':unittest.main()
