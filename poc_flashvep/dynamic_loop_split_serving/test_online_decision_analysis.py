import unittest
from .online_decision_analysis import evaluate


class OnlineDecisionAnalysisTests(unittest.TestCase):
    def test_proposal_rejection_and_actual_work_remain_distinct(self):
        cfg = dict(name='test', model='ouro', axis='loop', depth=4, length=16,
                   count=2, phase='CLEAN_FINITE_SERVING')
        events = [dict(event='ADMISSION_CALIBRATED_PLAN', time=2, request_id='r0_q0',
            online=True, selected_key=3, reason='PARTIAL'),
            dict(event='FRONTIER_CALIBRATED_RECHECK', time=3, request_id='r0_q0',
                online=True, selected_key=3, reason='SAFE_FULL_PD_FALLBACK'),
            dict(event='ADMISSION_CALIBRATED_PLAN', time=4, request_id='r0_q1',
                online=True, selected_key=4, reason='FULL_PD'),
            dict(event='FORCED_EXACT_FRONTIER_ACCEPTANCE', time=4, request_id='r0_q0', online=False)]
        parts = {1: dict(partial=False, assigned_D_recurrent_token_rows=0),
                 2: dict(partial=False, assigned_D_recurrent_token_rows=0)}
        row = evaluate(cfg, events, parts, {1, 2}, 1)
        self.assertEqual(row['partial_plans'], 1)
        self.assertEqual(row['frontier_rejects'], 1)
        self.assertEqual(row['completed_offloaded_serving_requests'], 0)
        parts[1] = dict(partial=True, assigned_D_recurrent_token_rows=16, accepted_cut=3)
        with self.assertRaisesRegex(AssertionError, 'acceptance'):
            evaluate(cfg, events, parts, {1, 2}, 1)
        events[1]['reason'] = 'CALIBRATED_FRONTIER_ACCEPT'
        row = evaluate(cfg, events, parts, {1, 2}, 1)
        self.assertEqual(row['completed_offloaded_serving_requests'], 1)
        self.assertEqual(row['accepted_partial_keys'], {'3': 1})
        self.assertEqual(row['assigned_D_recurrent_rows'], 16)


if __name__ == '__main__':
    unittest.main()
