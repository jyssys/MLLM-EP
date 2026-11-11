import json
import unittest
from .common import ROOT,digest
from .structural_analysis import pairs
from .confirmation_stats import paired_ratio
from .core_timing import supported_boundaries
from types import SimpleNamespace
from .quantum_contract import validate_spans


class Contracts(unittest.TestCase):
    def test_prelude_quantum_is_observed(self):
        cfg=dict(depth=32,length=2048,count=1,D_prefill_loop_cap=1,D_prefill_token_cap=None)
        alias=[dict(seed=1,phase='prefilling_prelude',count=64,loop_count=32)]
        self.assertFalse(validate_spans(cfg,alias)['passed'])
        proper=[dict(seed=1,phase='prefilling_prelude',count=64,loop_count=1)]
        self.assertTrue(validate_spans(cfg,proper)['passed'])
    def test_model_specific_observer(self):
        self.assertEqual(supported_boundaries(SimpleNamespace()),[])
        model=SimpleNamespace(prelude_prepared=lambda:None,coda_prepared=lambda:None)
        self.assertEqual([r[1] for r in supported_boundaries(model)],['prelude','prompt_coda'])
    def test_start_unit_bootstrap(self):
        value=paired_ratio([(9,10),(18,20),(27,30)])
        self.assertEqual(value['starts'],3)
        self.assertAlmostEqual(value['ratio'],.9)
        self.assertTrue(all(abs(x-.9)<1e-12 for x in value['CI95']))
        self.assertEqual(paired_ratio([(0,0)])['status'],'UNDEFINED_NONPOSITIVE_METRIC')

    def test_equal_rows_are_not_equal_attention(self):
        self.assertEqual(2048,64*32)
        self.assertEqual(pairs(0,2048),2098176)
        self.assertEqual(pairs(1984,64)*32,4129792)
        self.assertGreater(pairs(1984,64)*32,pairs(0,2048))

    def test_fixed_policy_heldout_confirmation(self):
        value=json.loads((ROOT/'analysis/fixed_policy_confirmation.json').read_text())
        self.assertEqual(value['policy_SHA256'],digest(ROOT/'configs/operator_kernel_policy_s1.json'))
        self.assertTrue(all(r['heldout_operator_exact'] for r in value['projections']))
        self.assertTrue(all(r['timing_source']!='selection_graph' for r in value['projections']))

    def test_symmetric_runtime_budget(self):
        for model in ('huginn','ouro'):
            configs=[json.loads((ROOT/'configs'/f'kernel_{model}_{arm}_state32.json').read_text())
                for arm in ('zero','loop','token')]
            for key in ('model_revision','cap','length','depth','gpus','P_slots','D_slots',
                    'kv_blocks','D_kv_blocks','deep_graph_budget','projection_kernel_policy_SHA256'):
                self.assertEqual(len({json.dumps(c[key]) for c in configs}),1,(model,key))
            self.assertEqual({c['deep_graph_budget'] for c in configs},{64})

    def test_search_budget(self):
        for model in ('huginn','ouro'):
            for start in (1,2,3):
                value=json.loads((ROOT/'raw'/f'gemm_{model}_search_s{start}/projection_results.json').read_text())
                self.assertEqual(value['config']['workspace_bytes'],64*1024**2)
                self.assertEqual(value['config']['maximum_candidates'],32)
                self.assertTrue(all(len(r['candidates'])<=32 for r in value['results']))


if __name__=='__main__':unittest.main()
