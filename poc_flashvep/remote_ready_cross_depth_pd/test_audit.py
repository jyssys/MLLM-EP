"""Guard against warmup, false readiness, and depth-label overclaims."""
import copy,json,unittest
from .audit import compatibility,selected_measured,remote_evidence_present,cpu_replay
from .common import ROOT


class TraceAuditTests(unittest.TestCase):
    def test_warmup_is_not_a_measured_transfer(self):
        batch=dict(host_begin=2,spans=[dict(request_id='warmup_tid')])
        self.assertFalse(selected_measured(batch,{'measured_tid':{}},dict(start=1,end=3)))

    def test_measurement_window_is_required(self):
        self.assertFalse(selected_measured(dict(host_begin=0,spans=[dict(request_id='tid')]),{'tid':{}},dict(start=1,end=3)))

    def test_cross_depth_only_is_not_an_aligned_pair(self):
        result=compatibility([dict(phase='decode',depth=0,rows=1),dict(phase='prefill',depth=20,rows=128)])
        self.assertTrue(result['cross_depth_only']);self.assertFalse(result['any_aligned_pair'])
        self.assertEqual(result['unmatched_prefill_rows'],128)

    def test_an_aligned_pair_does_not_make_the_entire_batch_single_depth(self):
        result=compatibility([dict(phase='decode',depth=0,rows=1),dict(phase='decode',depth=3,rows=1),dict(phase='prefill',depth=3,rows=128)])
        self.assertTrue(result['any_aligned_pair']);self.assertFalse(result['cross_depth_only'])
        self.assertFalse(result['strict_single_depth'])

    def test_without_decode_there_is_no_cross_phase_opportunity(self):
        self.assertFalse(compatibility([dict(phase='prefill',depth=20,rows=128)])['cross_depth_only'])

    def test_source_cut_and_exact_import_are_required(self):
        metrics={'tid':dict(timings=dict(d_received=1))};splits={'tid':dict(ownership='P_to_D')}
        checks={'tid':dict(passed=True,expected={'KV':1},actual={'KV':1})}
        self.assertTrue(remote_evidence_present('tid',metrics,splits,checks))
        self.assertFalse(remote_evidence_present('tid',metrics,{},checks))
        self.assertFalse(remote_evidence_present('unknown_generation',metrics,splits,checks))
        bad=copy.deepcopy(checks);bad['tid']['actual']['KV']=2
        self.assertFalse(remote_evidence_present('tid',metrics,splits,bad))

    def test_missing_receive_observation_is_not_remote_ready(self):
        self.assertFalse(remote_evidence_present('tid',{'tid':dict(timings={})},
            {'tid':dict(ownership='P_to_D')},{'tid':dict(passed=True,expected={},actual={})}))

    def test_current_native_policy_accepts_both_models_observed_depths(self):
        frames=json.loads((ROOT/'raw/selected_D_core_frames.json').read_text())
        examples=[next(r for r in frames if r['model']==m and r['remote_prefill_rows']) for m in ('ouro','huginn')]
        self.assertTrue(cpu_replay(examples)['passed'])

    def test_global_unused_opportunities_remain_unknown(self):
        data=json.loads((ROOT/'analysis/observed_opportunities.json').read_text())
        self.assertFalse(data['ready_queue_complete'])
        self.assertTrue(all(r['unused_legal_opportunities']=='NOT_MEASURED' for r in data['runs']))


if __name__=='__main__':unittest.main()
