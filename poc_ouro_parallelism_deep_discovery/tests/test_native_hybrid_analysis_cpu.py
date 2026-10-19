"""CPU checks for hybrid denominators, timestamps, and clean-lane admission."""
import copy
import unittest

from poc_ouro_parallelism_deep_discovery.scripts.analyze_native_hybrid_cohorts import hybrid_trial
from poc_ouro_parallelism_deep_discovery.tp2_dp2_cohort import aggregate, assignments


def fixture(count=2):
    case = dict(C=count, context=2, requests=[dict(cohort_slot=index,
        source_request_id=f'fixture{index}', prompt_token_ids=[7, 8]) for index in range(count)])
    rows = {}
    for worker, group in enumerate(assignments(case)):
        slots = [request['cohort_slot'] for request in group]
        rows[worker] = dict(worker=worker, physical_gpus=[4 + 2 * worker, 5 + 2 * worker],
            assigned_slots=slots, generated_tokens=len(slots) * 128,
            output_identity='ALL_FROZEN_CONSTANT_TOKEN_IDS', schedule_trace=None,
            release_received_ns=1100 + worker,
            local_generate_start_ns=1200 if slots else None,
            local_generate_end_ns=1001200 if slots else None,
            local_generate_wall_ns=1000000 if slots else 0,
            request_records=[dict(cohort_slot=request['cohort_slot'], source_request_id=request['source_request_id'],
                output_tokens=128, finish_reason='length') for request in group])
    raw = aggregate(case, rows, 128, 1000, 2001000)
    raw.update(size='1.4B', R=4, TP=2, DP=2, C=count, context=2, lane='CLEAN', quality='NOT_RUN',
               physical_gpus=[4, 5, 6, 7], physical_groups=[[4, 5], [6, 7]])
    return case, raw


class HybridAnalysisTests(unittest.TestCase):
    def test_observed_parent_wall_not_worker_max(self):
        case, raw = fixture()
        value = hybrid_trial(raw, '1.4B', case)
        self.assertEqual(value['wall_ms'], 2)
        self.assertEqual(value['longest_active_local_generate_ms'], 1)
        self.assertEqual(value['charged_IPC_controller_other_residual_ms'], 1)
        self.assertEqual(value['output_TPS'], 128000)

    def test_idle_replica_remains_charged_but_has_no_work(self):
        case, raw = fixture(1)
        self.assertEqual(raw['active_replicas'], 1)
        self.assertEqual(raw['reserved_gpus'], 4)
        self.assertEqual(hybrid_trial(raw, '1.4B', case)['output_TPS'], 64000)
        raw['workers'][1]['local_generate_wall_ns'] = 1
        with self.assertRaises(ValueError):
            hybrid_trial(raw, '1.4B', case)

    def test_identity_denominator_and_groups_fail_closed(self):
        for mutation in [lambda r: r.update(TP=4), lambda r: r.update(DP=4),
                         lambda r: r.update(size='2.6B'), lambda r: r.update(R=2),
                         lambda r: r.update(generated_tokens=128), lambda r: r.update(active_replicas=1),
                         lambda r: r.update(reserved_gpus=2), lambda r: r.update(cohort_output_TPS=256000),
                         lambda r: r.update(global_makespan_ms=float('nan')),
                         lambda r: r.update(physical_groups=[[4, 6], [5, 7]]),
                         lambda r: r['workers'][1].update(worker=0),
                         lambda r: r['workers'][0]['request_records'][0].update(source_request_id='wrong')]:
            case, raw = fixture()
            mutation(raw)
            with self.subTest(raw=raw), self.assertRaises(ValueError):
                hybrid_trial(raw, '1.4B', case)

    def test_clean_lane_has_no_trace_or_quality_claim(self):
        for mutation in [lambda r: r.update(lane='TRACE'), lambda r: r.update(quality='PASS'),
                         lambda r: r['workers'][0].update(schedule_trace=[])]:
            case, raw = fixture()
            mutation(raw)
            with self.assertRaises(ValueError):
                hybrid_trial(raw, '1.4B', case)

    def test_timestamp_and_output_endpoints_fail_closed(self):
        for mutation in [lambda r: r.update(global_makespan_ns=1),
                         lambda r: r['workers'][0].update(local_generate_start_ns=1099),
                         lambda r: r['workers'][0].update(local_generate_end_ns=3000000),
                         lambda r: r['workers'][0]['request_records'][0].update(output_tokens=127),
                         lambda r: r['workers'][0]['request_records'][0].update(finish_reason='eos')]:
            case, raw = fixture()
            mutation(raw)
            with self.assertRaises(ValueError):
                hybrid_trial(raw, '1.4B', case)


if __name__ == '__main__':
    unittest.main()
