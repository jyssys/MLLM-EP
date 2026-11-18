"""Test the exact arithmetic and the common budget predicate on CPU."""
import unittest
from .configuration import destination_limit, matched_token_plan


class ConfigurationTests(unittest.TestCase):
    def test_same_credit_lifecycle_includes_native_cut(self):
        import inspect
        import textwrap
        from ..dense_ar_slo_layer_pd import credits
        from .common_credits import transform
        scope=dict(credits.__dict__)
        exec(compile(transform(textwrap.dedent(inspect.getsource(credits.Credits.reserve))),
            'CPU_common_credit_contract','exec'),scope)
        for cut in (24,28,32,34,36):
            ledger=credits.Credits()
            self.assertFalse(scope['reserve'](ledger,'small',cut,['q0'],20,19))
            self.assertTrue(scope['reserve'](ledger,'g',cut,['q0'],20,1000))
            ledger.commit('g');ledger.map_request('internal','q0')
            ledger.allocated('internal');ledger.finish('q0')
            self.assertFalse(ledger.reservations)
    def test_chunk_adapter_transforms_only_expected_source_contracts(self):
        import inspect
        from ..dense_ar_slo_layer_pd import dynamic_extension
        from .segmented_source import transform_extension
        code=transform_extension(inspect.getsource(dynamic_extension.install))
        compile(code,'chunk_adapter_CPU_gate','exec')
        self.assertIn("assert role=='prefill' or all",code)
        self.assertIn('continuation_credits=',code)
    def test_exact_rows_all_cuts(self):
        for m in (2,4,8,12):
            plan=matched_token_plan(m,144,group_size=1)
            self.assertEqual(sum(plan)*36,8000*m*144)
            self.assertTrue(all((8000-s)%16==0 for s in plan))

    def test_128_cannot_match_all_rows(self):
        for m in (2,4,8,12):
            with self.assertRaises(AssertionError):
                matched_token_plan(m,128,group_size=1)

    def test_same_admission_rule_different_query_geometry(self):
        common=dict(count=8,D_token_budget=8192,length=8000)
        layer=dict(common,policy='STATIC')
        token=dict(common,policy='NATIVE',token_axis=True,token_suffix=2672)
        native=dict(common,policy='NATIVE')
        self.assertEqual(destination_limit(list(range(8)),layer,8),1)
        self.assertEqual(destination_limit(list(range(8)),token,8),3)
        self.assertEqual(destination_limit(list(range(8)),native,8),8)

    def test_last2_supplement_has_no_original_matrix_overlap(self):
        from .campaign import matrix
        from .last2_supplement import matrix as supplement
        original=matrix();added=supplement()
        self.assertEqual(len(original),232)
        self.assertEqual(len(added),88)
        self.assertFalse({c['name'] for c in original}&{c['name'] for c in added})
        self.assertEqual(sum(c['matrix_stage']=='confirmation' for c in added),30)
        native={(c['cap'],c['arrival_rate'],c['arrival_seed']):c for c in original
            if c['evaluation_arm']=='Native'}
        for cfg in added:
            self.assertEqual(cfg['static_layers'],2)
            self.assertEqual(cfg['layer_cut'],34)
            self.assertEqual(cfg['D_admission_batch'],8)
            self.assertEqual(cfg['token_budget'],8192)
            self.assertEqual(cfg['D_token_budget'],8192)
            self.assertEqual(cfg['P_slots'],1024)
            self.assertEqual(cfg['D_slots'],1024)
            self.assertTrue(cfg['common_prospective_KV_reservation'])
            if cfg['evaluation_arm']=='Token':
                self.assertEqual(36*sum(cfg['token_suffix_by_request']),144*8000*2)
                self.assertIsNone(cfg['layer_mode'])
                self.assertEqual(cfg['policy'],'NATIVE')
                self.assertEqual(destination_limit(list(range(8)),cfg,8),8)
            if cfg['matrix_stage']!='confirmation':
                reference=native[cfg['cap'],cfg['arrival_rate'],cfg['arrival_seed']]
                for field in ('arrival_offsets','dataset','request_caps','gpus'):
                    self.assertEqual(cfg[field],reference[field])

    def test_last2_qualification_uses_fresh_native_partition_hashes(self):
        from .last2_supplement import state_configuration
        for method in ('Native','Layer','Token'):
            cfg=state_configuration(method,8)
            self.assertEqual(cfg['witness_hidden_partition_tokens'],[432,448])
            self.assertEqual(cfg['D_admission_batch'],8)
            self.assertTrue(cfg['correctness_only'])
            if method=='Token':
                self.assertEqual(cfg['token_suffix_by_request'],[432,448]*4)


if __name__=='__main__':
    unittest.main()
