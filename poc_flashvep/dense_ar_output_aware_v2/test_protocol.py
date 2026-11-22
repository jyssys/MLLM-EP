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
    def test_unknown_runtime_switch_needs_new_proof(self):
        a=dict(method='Layer',axis_cut=8,P_compile_prefix=False,name='q1')
        b=dict(a,P_compile_prefix=True,name='q2')
        self.assertNotEqual(signature(a),signature(b))
        self.assertEqual(signature(a),signature(dict(a,name='q3',qps=3)))
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

if __name__=='__main__':unittest.main()
