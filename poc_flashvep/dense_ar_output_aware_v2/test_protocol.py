"""CPU regressions for the user's winner and protection contracts."""
import unittest
from .statistics import ratio,point_winner,confidence,safety,ratio_of_ratios
from .runner import signature
from .dataset import bucket

class ProtocolTests(unittest.TestCase):
    def test_output_bins(self):
        self.assertEqual([bucket(x) for x in (0,1,10,11,50,51,100,101,200,201)],
            [None,0,0,1,1,2,2,3,3,None])
    def test_point_does_not_hide_uncertainty(self):
        r=ratio([80,100,110],[100,100,100])
        self.assertEqual(point_winner(r),'LAYER_POINT_WIN')
        self.assertEqual(confidence(r),'TIE_OR_INCONCLUSIVE')
    def test_ratio_of_start_means_not_mean_ratios(self):
        self.assertAlmostEqual(ratio([2,4,8],[1,2,8])['estimate'],14/11)
    def test_single_start_has_no_CI(self):
        self.assertIsNone(ratio([1],[2])['CI95'])
    def test_output_endpoint_contrast_uses_ratios_of_means(self):
        r=ratio_of_ratios([2,4,8],[1,2,8],[3,6,12],[3,3,3])
        self.assertAlmostEqual(r['estimate'],(14/11)/(21/9))
        self.assertIsNotNone(r['CI95'])
    def test_three_axis_exact_point_tie_is_not_lexicographic(self):
        from .c3_subgroups import point_winner as three_axis_winner
        self.assertEqual(three_axis_winner(dict(Layer=1,StockToken=1,Hybrid=2)),'EQUAL')
        self.assertEqual(three_axis_winner(dict(Layer=1,StockToken=2,Hybrid=3)),'Layer')
    def test_paired_stage_difference_preserves_sign(self):
        from .statistics import paired_difference
        result=paired_difference([1,2,3],[4,5,6])
        self.assertEqual(result['estimate'],-3)
        self.assertEqual(result['CI95'],[-3,-3])
    def test_host_queue_quantile_does_not_interpolate_a_fake_count(self):
        from .queue_analysis import weighted_states
        rows=[dict(time=0,waiting_total=0),dict(time=5,waiting_total=10)]
        value=weighted_states(rows,0,10,'waiting_total')
        self.assertEqual(value['time_weighted_mean'],5)
        self.assertEqual(value['time_weighted_p95'],10)
    def test_replay_is_not_relabelled_as_an_ordinary_fixed_output_run(self):
        from .derived_metrics import output_contract
        self.assertEqual(output_contract(dict(ignore_eos=False)),'NATURAL_EOS')
        self.assertEqual(output_contract(dict(ignore_eos=True)),'CONTROLLED_FIXED_OUTPUT')
        self.assertEqual(output_contract(dict(ignore_eos=True,diagnostic_controlled_output_replay=True,replay_output_lengths=[5])),
            'DIAGNOSTIC_FIXED_LENGTH_REPLAY')
    def test_main_knee_worklists_preserve_all_jobs_and_pair_identity(self):
        import json,tempfile
        from pathlib import Path
        from unittest.mock import patch
        from . import c3_main
        from .common import write
        with tempfile.TemporaryDirectory(prefix='v2-phased-plan-test-') as folder:
            root=Path(folder);write(root/'configs/c3_budget_priority_amendment_v4.json',dict(primary_confirmation_QPS=2.0))
            plans=[]
            for heldout in (False,True):
                blocks=[]
                for qps in (2.0,4.0):
                    lane=1 if heldout and qps==2 else 2 if heldout else 0 if qps==2 else 1
                    for start in range(3):
                        names={label:f'{heldout}_{qps}_{start}_{label}' for label in ('Layer','StockToken')}
                        for name in names.values():write(root/'configs'/(name+'.json'),dict(gpus=[2*lane,2*lane+1]))
                        blocks.append(dict(qps=qps,start=start,pair=[2*lane,2*lane+1],arms=names))
                filename='heldout.json' if heldout else 'main.json';write(root/'jobs'/filename,dict(blocks=blocks));plans.append(filename)
            with patch.object(c3_main,'ROOT',root):first,second=c3_main.phased_worklists(*plans)
            a=json.loads((root/'jobs'/first).read_text());b=json.loads((root/'jobs'/second).read_text())
            aa={n for lane in a['lanes'] for n in lane};bb={n for lane in b['lanes'] for n in lane}
            self.assertFalse(aa&bb);self.assertEqual(len(aa|bb),24)
            self.assertEqual(sum(x['qps']==2 for x in a['blocks']),6)
            self.assertEqual(sum(x['qps']==4 and x['start']==0 for x in a['blocks']),1)
            self.assertEqual(len(a['lanes'][2]),0)
    def test_unknown_runtime_switch_needs_new_proof(self):
        a=dict(method='Layer',axis_cut=8,P_compile_prefix=False,name='q1')
        b=dict(a,P_compile_prefix=True,name='q2')
        self.assertNotEqual(signature(a),signature(b))
        self.assertEqual(signature(a),signature(dict(a,name='q3',qps=3)))
        self.assertNotEqual(signature(a),signature(dict(a,D_compile_partial=True)))
    def test_both_native_references_are_required(self):
        row=dict(ITL_p99_ms=8,TPOT_p95_ms=8,decode_SLO_attainment=1,every_ITL_SLO_attainment=1,max_ITL_ms=9)
        self.assertFalse(safety([row]*3,{'NativeStrong':[row]*3})['qualified'])
        self.assertTrue(safety([row]*3,{'NativeStrong':[row]*3,'NativeCommon':[row]*3})['qualified'])
    def test_common_tail_failure_cannot_hide_behind_Strong(self):
        row=dict(ITL_p99_ms=9,TPOT_p95_ms=8,decode_SLO_attainment=1,every_ITL_SLO_attainment=1,max_ITL_ms=12)
        slow=dict(row,ITL_p99_ms=15);common=dict(row,ITL_p99_ms=8)
        self.assertFalse(safety([row]*3,{'NativeStrong':[slow]*3,'NativeCommon':[common]*3})['qualified'])
    def test_stock_current_dataset_map_at_actual_configuration_seam(self):
        import copy,json
        from .common import ROOT
        from .work_contract import regenerate,validate
        old=json.loads((ROOT/'configs/V2_C3Q_StockToken8_Natural400.json').read_text())
        rows=json.loads((ROOT/'data'/old['dataset']).read_text())
        with self.assertRaises(AssertionError):validate(old,rows)
        fixed=regenerate(copy.deepcopy(old),rows)
        self.assertTrue(validate(fixed,rows))
        self.assertNotEqual(old['suffix_by_request'],fixed['suffix_by_request'])
    def test_actual_C0_recorded_token_layer_work_is_exact(self):
        import json
        from .common import ROOT
        from .work_contract import executed_prompt_work
        for axis in ('Layer','Token'):
            name=f'V2_C0_discovery_I0_O8_C2_s0_{axis}'
            cfg=json.loads((ROOT/'configs'/(name+'.json')).read_text())
            rows=json.loads((ROOT/'data'/cfg['dataset']).read_text())
            value=executed_prompt_work(cfg,ROOT/'raw'/name,rows)
            self.assertEqual(value['actual_P_minus_nominal_rows'],0)
            self.assertEqual(value['actual_D_minus_nominal_rows'],0)
            self.assertEqual(value['observed_total_minus_ideal_rows'],0)

if __name__=='__main__':unittest.main()
