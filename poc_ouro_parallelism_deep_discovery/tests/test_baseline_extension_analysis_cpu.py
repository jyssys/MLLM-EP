"""CPU denominator and source-counter admission checks for baseline scoring."""
import copy
import unittest

from poc_ouro_parallelism_deep_discovery.contract import MODELS
from poc_ouro_parallelism_deep_discovery.scripts.analyze_baseline_extension import request_metrics, domain_summary


def evaluator(row, result):
    return dict(answer_matches=True, parsed=True, truncated=True, correct_completed=False)


def fixture():
    row = dict(id='fixture/1', dataset='gsm8k', answer='1')
    value = dict(id='fixture/1', dataset='gsm8k', size='1.4B', mode='adaptive', physical_gpu=5,
        revision=MODELS['1.4B']['revision'], source_protocol_sha256='HASH', cap=8192, temperature=0, EOS=0,
        evidence_class='QUALITY_ONLY', output_ids=[7, 8, 9, 0], generated_tokens=4, prompt_ids=[1] * 10,
        depth_events=[dict(position=10 + index, depth=depth) for index, depth in enumerate([2, 3, 4])],
        depth_count_matches=True, depth_range_matches=True, finish_reason='eos', text='1', wall_ms=10,
        stats=dict(generated_tokens=4, recurrent_steps=9, recurrent_batches=9, coda_tokens=3,
            coda_batches=3, fixed_depth_recurrent_steps=12, prefill_batches=1, prefill_requests=1,
            prefill_coda_tokens=1, prefill_recurrent_steps=40, exit_depth_histogram={'1': 1, '2': 1, '3': 1},
            recurrent_graph_hits=8, recurrent_graph_captures=1, scheduler_refills=0, mixed_recurrent_batches=0))
    return row, value


class BaselineAnalysisTests(unittest.TestCase):
    def test_source_counters_and_decode_only_denominator(self):
        row, value = fixture()
        result = request_metrics(row, value, '1.4B', 'adaptive', 5, 'HASH', evaluator)
        self.assertEqual(result['recorded_decode_core_pass_work'], 9)
        self.assertEqual(result['recorded_decode_positions'], 3)
        self.assertEqual(result['mean_decode_depth'], 3)
        self.assertEqual(result['physical_layer_execution_proxy'], 216)
        self.assertEqual(result['normalized_recorded_decode_work'], 0.75)
        self.assertTrue(result['source_counters_match'])

    def test_frozen_stop_counts_completed_not_truncated(self):
        row, value = fixture()
        value['output_ids'][-1] = 11
        value['finish_reason'] = 'stop'
        result = request_metrics(row, value, '1.4B', 'adaptive', 5, 'HASH', evaluator)
        self.assertTrue(result['correct_completed'])
        self.assertFalse(result['truncated'])

    def test_identity_policy_and_nonfinite_wall_fail_closed(self):
        for mutation in [lambda r: r.update(id='wrong'), lambda r: r.update(physical_gpu=4),
                         lambda r: r.update(mode='fixed'), lambda r: r.update(temperature=1),
                         lambda r: r.update(cap=64), lambda r: r.update(revision='wrong'),
                         lambda r: r.update(wall_ms=float('nan')), lambda r: r.update(generated_tokens=3),
                         lambda r: r.update(depth_count_matches=False), lambda r: r.update(source_protocol_sha256='wrong')]:
            row, value = fixture()
            mutation(value)
            with self.assertRaises(ValueError):
                request_metrics(row, value, '1.4B', 'adaptive', 5, 'HASH', evaluator)

    def test_positions_native_work_and_graph_coverage_fail_closed(self):
        for mutation in [lambda r: r['depth_events'][1].update(position=10),
                         lambda r: r['depth_events'][0].update(position=9),
                         lambda r: r['depth_events'][0].update(depth=1),
                         lambda r: r['stats'].update(recurrent_steps=8),
                         lambda r: r['stats'].update(prefill_recurrent_steps=10),
                         lambda r: r['stats'].update(recurrent_graph_hits=9),
                         lambda r: r['stats'].update(exit_depth_histogram={'1': 3}),
                         lambda r: r['stats'].update(scheduler_refills=1)]:
            row, value = fixture()
            mutation(value)
            with self.assertRaises(ValueError):
                request_metrics(row, value, '1.4B', 'adaptive', 5, 'HASH', evaluator)

    def test_request_weighted_and_token_weighted_depth_are_separate(self):
        row, value = fixture()
        a = request_metrics(row, value, '1.4B', 'adaptive', 5, 'HASH', evaluator)
        b = copy.deepcopy(a)
        a.update(id='a', recorded_decode_positions=1, recorded_decode_core_pass_work=4, mean_decode_depth=4,
                 normalized_recorded_decode_work=1, R2_positions=0, R3_positions=0, R4_positions=1)
        b.update(id='b', recorded_decode_positions=3, recorded_decode_core_pass_work=6, mean_decode_depth=2,
                 normalized_recorded_decode_work=0.5, R2_positions=3, R3_positions=0, R4_positions=0)
        result = domain_summary([a, b])
        self.assertEqual(result['request_weighted_mean_decode_depth'], 3)
        self.assertEqual(result['token_weighted_mean_decode_depth'], 2.5)
        self.assertEqual(result['request_weighted_normalized_decode_work'], 0.75)
        self.assertEqual(result['N'], 2)
        self.assertEqual(result['accuracy_CI_percent'], [100, 100])


if __name__ == '__main__':
    unittest.main()
