import unittest
from poc_native_lp_ep4.scripts.aggregate import component_waves
from poc_native_lp_ep4.scripts.evidence import restart_estimate, restart_sensitivity


def event(histogram,frame):
    return {'run':'unit','wave':0,'arm':'CP512','modality':'VISUAL_PREFILL',
        'repeat':0,'layer':0,'iteration':frame,'prefill_active':True,'ranks':4,
        'rank_durations':{str(r):{'dispatch_ms':10 if r==0 else 1,
            'expert_execution_ms':1 if r==0 else 10,'combine_ms':1,'moe_ms':12} for r in range(4)},
        'dispatch_ms':10,'loaded_expert_weight_bytes_per_expert':9437184,
        'injected_histogram':histogram,
        'logical_injected_dispatch_matrix':[[0,4096,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0]]}


class EvidenceTest(unittest.TestCase):
    def test_restart_bootstrap_keeps_all_repeats(self):
        estimate=restart_estimate([0,0,0,10],['a','a','a','b'])
        self.assertEqual(estimate['mean'],2.5)
        self.assertEqual(estimate['n_restarts'],2)
        self.assertEqual(estimate['repeat_counts_by_restart'],[3,1])
        self.assertIsNone(restart_estimate([1],['one'])['ci95_restart'])

    def test_restart_pairs_cannot_cross_runs(self):
        rows=[{'run':'first','repeat':0,'arm':'FULL','modality':'VISUAL_PREFILL','ttft_ms':100},
              {'run':'second','repeat':0,'arm':'CP512','modality':'VISUAL_PREFILL','ttft_ms':200}]
        out=restart_sensitivity(rows)
        self.assertFalse(any(r['comparison']=='PAIRED_POLICY_DELTA' for r in out))

    def test_reload_amp_counts_reactivation_not_more_tokens(self):
        hs=[1,1]+[0]*126
        out=component_waves([event(hs,0),event(hs,1)])[0]
        self.assertEqual(out['expert_reload_amp'],2)
        self.assertEqual(out['logical_remote_dispatch_bytes'],8192)
        self.assertEqual(out['expert_weight_bytes_proxy'],4*9437184)

    def test_components_stay_on_one_rank_not_sum_independent_maxima(self):
        out=component_waves([event([2]+[0]*127,0)])[0]
        self.assertEqual(out['moe_total_ms'],12)
        self.assertEqual(out['dispatch_total_ms']+out['expert_total_ms']+out['combine_total_ms'],12)
        self.assertEqual(out['max_rank_expert_total_ms'],10)

    def test_missing_route_histogram_is_not_zero_traffic(self):
        row=event(None,0)
        out=component_waves([row])[0]
        self.assertIsNone(out['expert_reload_amp'])
        self.assertIsNone(out['logical_remote_dispatch_bytes'])


if __name__=='__main__':unittest.main()
