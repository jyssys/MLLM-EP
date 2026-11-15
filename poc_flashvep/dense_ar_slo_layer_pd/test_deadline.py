"""Test the actual affected-work predicate, including retirement semantics."""
import unittest
from .deadline import affected_deadline_conflicts


class DeadlineTests(unittest.TestCase):
    def test_ready_decode_is_delayed(self):
        work=[dict(request_id='q1',kind='ITL',ready_at=1.,deadline=1.03,service_ms=20.)]
        conflicts,_=affected_deadline_conflicts(1.,1.02,work)
        self.assertEqual(len(conflicts),1)
        self.assertAlmostEqual(conflicts[0]['predicted_lateness_ms'],10.)
    def test_quantum_fits_before_deadline(self):
        work=[dict(request_id='q1',kind='ITL',ready_at=1.,deadline=1.05,service_ms=20.)]
        self.assertEqual(affected_deadline_conflicts(1.,1.02,work)[0],[])
    def test_retired_cohort_does_not_create_false_deadline(self):
        work=[dict(request_id='q1',kind='ITL',ready_at=.9,deadline=.93,retire_upper=.98)]
        conflicts,retired=affected_deadline_conflicts(1.,1.2,work)
        self.assertEqual(conflicts,[]);self.assertEqual(retired,['q1'])
    def test_future_ready_after_suffix_is_not_affected(self):
        work=[dict(request_id='q1',kind='ITL',ready_at=1.3,deadline=1.33)]
        self.assertEqual(affected_deadline_conflicts(1.,1.2,work)[0],[])
    def test_nominal_objective_does_not_change_safety_predictions(self):
        from .common import ROOT,json
        from .configuration import base
        from .method_discovery_v2 import method_settings
        from .policy import LayerPolicy,Observation
        cfg=base('CPU_NOMINAL_FIXTURE','DIRECT',count=8,cap=16,rate=4.8,**method_settings())
        costs=json.loads((ROOT/'configs/coalesced_cost_model_v2.json').read_text())
        state=Observation(now=100.,group_size=8,P_backlog=16,D_available=101.,
            D_available_nominal=100.4,pending_requests=0,KV_free_blocks=20000,
            active_decode=4,maximum_group_output_tokens=16,input_token_lengths=(8000,)*8)
        first=LayerPolicy(cfg,costs).choose(state)
        second=LayerPolicy(dict(cfg,separate_nominal_objective=True),costs).choose(state)
        self.assertEqual(len(first.candidates),4)
        for a,b in zip(first.candidates,second.candidates):
            for key in ('predicted_D_finish','predicted_D_start','D_available_upper',
                        'safety_margin_ms','affected_deadline_conflicts',
                        'next_ingress_deadline_conflicts','own_TTFT_deadline_conflicts'):
                self.assertEqual(a[key],b[key],key)
            self.assertLessEqual(b['predicted_D_finish_nominal'],b['predicted_D_finish'])
    def test_no_queue_option_requires_personal_benefit_without_invented_pressure(self):
        from .common import ROOT,json
        from .configuration import base
        from .method_discovery_v2 import method_settings
        from .policy import LayerPolicy,Observation
        cfg=base('CPU_NO_QUEUE_FIXTURE','DIRECT',count=8,cap=16,rate=4.8,**method_settings())
        cfg.update(separate_nominal_objective=True)
        costs=json.loads((ROOT/'configs/coalesced_cost_model_v2.json').read_text())
        state=Observation(now=100.,group_size=8,P_backlog=0,D_available=100.,
            D_available_nominal=100.,pending_requests=0,KV_free_blocks=20000,
            active_decode=0,maximum_group_output_tokens=16,input_token_lengths=(8000,)*8)
        self.assertEqual(LayerPolicy(cfg,costs).choose(state).reason,'NO_P_COMPUTE_BACKLOG')
        choice=LayerPolicy(dict(cfg,allow_zero_queue_personal_benefit=True),costs).choose(state)
        self.assertGreater(choice.layers,0);self.assertEqual(state.P_backlog,0)
        selected=next(c for c in choice.candidates if c['layers']==choice.layers)
        self.assertEqual(selected['predicted_queue_saving_ms'],0.)
        self.assertGreater(selected['predicted_personal_nominal_gain_ms'],0.)
    def test_incremental_objective_keeps_upper_safety_tests(self):
        from .common import ROOT,json
        from .configuration import base
        from .method_discovery_v2 import method_settings
        from .policy import LayerPolicy,Observation
        cfg=base('CPU_INCREMENTAL_FIXTURE','DIRECT',count=8,cap=16,rate=4.8,**method_settings())
        cfg.update(separate_nominal_objective=True)
        costs=json.loads((ROOT/'configs/coalesced_cost_model_v2.json').read_text())
        state=Observation(now=100.,group_size=8,P_backlog=8,P_ready_external=0,
            P_ready_inside_macro=4,D_available=101.,D_available_nominal=100.4,
            pending_requests=0,KV_free_blocks=20000,active_decode=4,
            maximum_group_output_tokens=16,input_token_lengths=(8000,)*8)
        first=LayerPolicy(cfg,costs).choose(state)
        second=LayerPolicy(dict(cfg,pipeline_objective_excludes_existing_wait=True,
            queue_benefit_external_only=True),costs).choose(state)
        for a,b in zip(first.candidates,second.candidates):
            for key in ('predicted_D_finish','predicted_D_start','D_available_upper',
                        'affected_deadline_conflicts','next_ingress_deadline_conflicts',
                        'own_TTFT_deadline_conflicts','safety_margin_ms'):
                self.assertEqual(a[key],b[key],key)
            self.assertEqual(b['predicted_queue_saving_ms'],0.)

    def test_expired_ttft_recovery_does_not_relax_live_decode_deadlines(self):
        from dataclasses import replace
        from .common import ROOT,json
        from .configuration import base
        from .method_discovery_v2 import method_settings
        from .policy import LayerPolicy,Observation
        cfg=base('CPU_EXPIRED_TTFT_FIXTURE','DIRECT',count=8,cap=16,rate=4.8,**method_settings())
        cfg.update(separate_nominal_objective=True)
        costs=json.loads((ROOT/'configs/coalesced_cost_model_v2.json').read_text())
        state=Observation(now=100.,group_size=8,P_backlog=16,D_available=100.,
            D_available_nominal=100.,pending_requests=0,KV_free_blocks=20000,
            active_decode=0,maximum_group_output_tokens=16,input_token_lengths=(8000,)*8,
            group_request_ids=tuple('q'+str(i) for i in range(8)),
            own_TTFT_deadlines=(99.,)*8,
            affected_deadlines=(dict(request_id='q8',kind='TTFT',ready_at=100.,
                deadline=99.,ordering='AFTER_CANDIDATE',service_ms=20.),))
        strict=LayerPolicy(cfg,costs).choose(state)
        self.assertEqual(strict.layers,0)
        revised=dict(cfg,expired_ttft_policy='QUEUE_RECOVERY')
        recovered=LayerPolicy(revised,costs).choose(state)
        self.assertGreater(recovered.layers,0)
        self.assertTrue(all(c['expired_own_TTFT_not_protected'] for c in recovered.candidates))
        live=replace(state,own_TTFT_deadlines=(100.01,)*8)
        future=LayerPolicy(revised,costs).choose(live)
        self.assertEqual(future.layers,0)
        self.assertTrue(all('OWN_TTFT_DEADLINE' in c['rejection_reasons'] for c in future.candidates))
        active=replace(state,affected_deadlines=(dict(request_id='q8',kind='ITL',ready_at=100.,
            deadline=99.,retire_upper=10000.,service_ms=20.),))
        decode=LayerPolicy(revised,costs).choose(active)
        self.assertEqual(decode.layers,0)
        self.assertTrue(all('AFFECTED_DECODE_OR_FIRST_TOKEN_DEADLINE' in c['rejection_reasons']
            for c in decode.candidates))


if __name__=='__main__':unittest.main()
