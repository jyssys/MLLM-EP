import collections
import dataclasses
import json
import unittest
from poc_decode_first_depth_aware_dp.contract import ROOT,read,sha
from poc_decode_first_depth_aware_dp.runtime import stats_dict


class DecodeFirstContract(unittest.TestCase):
    def test_authoritative_spec_is_unchanged(self):
        self.assertEqual(sha(ROOT/'reports/spec.md'),read(ROOT/'artifacts/preregistration.json')['spec_sha256'])

    def test_policy_was_fixed_before_quality(self):
        registration=read(ROOT/'artifacts/preregistration.json')
        self.assertEqual(registration['primary_gate']['exit_threshold'],.5)
        self.assertEqual(registration['primary_gate']['min_recurrent_steps'],2)
        self.assertFalse(registration['RL'])
        rows=read(ROOT/'artifacts/quality_protocol_v2.json')['rows']
        pilot=[r for r in rows if r['split']=='pilot']
        self.assertEqual(collections.Counter(r['dataset'] for r in pilot),{'gsm8k':16,'math500':16})
        self.assertEqual(len({r['id'] for r in rows}),1819)

    def test_counter_serialization_preserves_values(self):
        @dataclasses.dataclass
        class Stats:
            histogram:collections.Counter=dataclasses.field(default_factory=collections.Counter)
        source=Stats(collections.Counter({1:2,3:11}))
        broken=dataclasses.asdict(source)
        with self.assertRaises(TypeError):json.dumps(broken)
        self.assertEqual(json.loads(json.dumps(stats_dict(source)))['histogram'],{'1':2,'3':11})

    def test_actual_core_skipping_and_KV_checks(self):
        fixed=read(ROOT/'outputs/runtime_smoke_v6/q_None.json')
        dynamic=read(ROOT/'outputs/runtime_smoke_v6/q_0.5.json')
        self.assertEqual(fixed['stats']['prefill_recurrent_steps'],52)
        self.assertEqual(dynamic['stats']['prefill_recurrent_steps'],52)
        self.assertEqual(fixed['stats']['recurrent_steps'],60)
        self.assertEqual(dynamic['stats']['recurrent_steps'],44)
        self.assertEqual(fixed['actual_layer_calls']-dynamic['actual_layer_calls'],384)
        self.assertEqual(dynamic['exit_copy_checks'],360)
        self.assertTrue(dynamic['all_exit_copies_identical'])

    def test_no_reference_failure_is_hidden(self):
        reference=read(ROOT/'outputs/reference_smoke_v1/reference_comparison.json')
        for name,limit in reference['limits'].items():self.assertLessEqual(reference[name],limit)
        self.assertFalse(reference['natural_ids_equal'])
        self.assertEqual(reference['top1_matches'],15)
        self.assertFalse(read(ROOT/'artifacts/quality_pilot_preflight.json')['scheduling_or_oracle_claim_allowed'])

    def test_every_recorded_GPU_job_has_an_exact_allowed_mask(self):
        paths=list((ROOT/'outputs').glob('*/run_contract.json'))
        self.assertGreaterEqual(len(paths),10)
        for p in paths:self.assertLessEqual(set(map(int,read(p)['mask'].split(','))),{4,5,6,7})

    def test_extension_preserves_frozen_order_and_prompts(self):
        original=read(ROOT/'artifacts/quality_protocol_v2.json')
        extended=read(ROOT/'artifacts/quality_protocol_v3_pilot32.json')
        self.assertTrue(extended['gate_unchanged'])
        self.assertEqual(len(extended['extension_ids']),32)
        old={r['id']:r for r in original['rows']}
        for row in extended['rows']:
            self.assertEqual(row['prompt'],old[row['id']]['prompt'])
            self.assertEqual(row['split']=='pilot',row['order']<32)

    def test_zero_lower_bound_does_not_trigger_severe_stop(self):
        from poc_decode_first_depth_aware_dp.scripts.quality_analyze import severe_pilot_stop
        self.assertFalse(severe_pilot_stop(18.75,[0.,37.5]))
        self.assertFalse(severe_pilot_stop(12.5,[1.,25.]))
        self.assertTrue(severe_pilot_stop(18.75,[3.125,34.375]))

    def test_measured_primary_quality_blocks_scheduling(self):
        summary=read(ROOT/'artifacts/quality_pilot_analysis_v2/quality_summary.json')
        self.assertEqual(summary['verdict'],'QUALITY_FAIL')
        self.assertFalse(summary['subsequent_stages_allowed'])
        math=next(r for r in summary['rows'] if r['dataset']=='math500')
        self.assertEqual((math['fixed_correct'],math['dynamic_correct'],math['n']),(24,19,32))
        self.assertEqual(math['paired_bootstrap_CI95_drop_pp'],[3.125,28.125])
        self.assertTrue(math['severe_pilot_failure'])
        final=read(ROOT/'artifacts/final_decision.json')
        self.assertEqual(final['quality_summary_sha256'],sha(ROOT/'artifacts/quality_pilot_analysis_v2/quality_summary.json'))
        self.assertIsNone(final['oracle_incremental_depth_gain'])
        self.assertTrue(final['no_RL'])

    def test_paired_cap_followup_preserves_primary_outputs(self):
        sensitivity=read(ROOT/'artifacts/quality_cap_sensitivity.json')
        self.assertTrue(sensitivity['primary_analysis_preserved'])
        self.assertTrue(sensitivity['all_primary_token_prefixes_equal'])
        self.assertTrue(sensitivity['all_primary_depth_prefixes_equal'])
        self.assertEqual(len(sensitivity['comparisons']),8)
        for row in sensitivity['comparisons']:
            self.assertEqual(row['primary_answer_matches'],row['followup_answer_matches'])
        self.assertFalse(sensitivity['subsequent_stages_allowed'])


if __name__=='__main__':unittest.main()
