import unittest
from pathlib import Path

from poc_native_lp_ep4.scripts.metrics import policy_deltas, modality_component_deltas


class PairedAnalysisTest(unittest.TestCase):
    def test_restart_plan_cap_and_ten_global_repeats(self):
        for arms in (3,4):
            per_block=48//(arms*4)-1
            offset=0;covered=[]
            while offset<10:
                count=min(per_block,10-offset)
                self.assertLessEqual(arms*4*(count+1),48)
                covered.extend(range(offset,offset+count));offset+=count
            self.assertEqual(covered,list(range(10)))

    def test_atomic_admission_uses_installed_api_outside_decode_window(self):
        source=Path('poc_native_lp_ep4/scripts/run.py').read_text()
        pause=source.index("call_utility('pause_scheduler','keep',False)")
        add=source.index('resident_ids=llm._add_completion_requests')
        resume=source.index("call_utility('resume_scheduler')")
        step=source.index('while llm.llm_engine.has_unfinished_requests()')
        self.assertTrue(pause<add<resume<step)

    def test_policy_delta_pairs_repeat_and_modality_not_layer_events(self):
        rows=[]
        for repeat in (0,1):
            for arm,value in (('FULL',100+repeat*100),('LP_G16',90+repeat*100)):
                rows.append({'arm':arm,'modality':'VISUAL_PREFILL','repeat':repeat,
                    'ttft_ms':value,'peak_itl_ms':value,'peak_vsel_ms':value,
                    'peak_itl_ratio':value,'p99_itl_ms':value,'lm_start_to_first_token_ms':None})
        result=next(r for r in policy_deltas(rows) if r['arm']=='LP_G16' and r['control']=='FULL')
        self.assertEqual(result['pairs'],2)
        self.assertEqual(result['ttft_ms_delta']['mean'],-10)
        self.assertEqual(result['ttft_ms_delta']['ci95'],[-10.,-10.])
        self.assertIsNone(result['lm_start_to_first_token_ms_delta'])

    def test_component_modality_delta_requires_both_observed_conditions(self):
        row={'arm':'LP_AUTO','repeat':0,'modality':'VISUAL_PREFILL',
            **{k:1. for k in ('dispatch_total_ms','expert_total_ms','combine_total_ms',
                            'moe_total_ms','max_dispatch_quantum_ms')}}
        self.assertEqual(modality_component_deltas([row]),{})


if __name__=='__main__':unittest.main()
