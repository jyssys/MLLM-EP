"""Exercise a local suffix frontier despite misleading imported counters."""
from types import SimpleNamespace
import unittest
from .fixed_matched_cohort_v2 import SubmissionFrontier


class LocalSuffixFrontierTests(unittest.TestCase):
    def test_online_matrix_keeps_both_axes_at_the_same_static_batches(self):
        from .fixed_cohort_online import matrix
        values=matrix()
        self.assertEqual(len(values),42)
        self.assertEqual(len({c['name'] for c in values}),42)
        paired={}
        for cfg in values:
            self.assertFalse(cfg['correctness_only'])
            self.assertEqual(cfg['P_group_size'],1)
            self.assertEqual(cfg['D_admission_batch'],2)
            self.assertEqual(cfg['matched_D_cohort_size'],2)
            self.assertEqual(cfg['token_budget'],8192)
            self.assertEqual(cfg['D_token_budget'],8192)
            self.assertEqual(cfg['P_slots'],1024)
            self.assertEqual(cfg['D_slots'],1024)
            self.assertEqual(cfg['cap'],16)
            key=(cfg['matrix_stage'],cfg['count'],cfg['arrival_rate'],cfg['arrival_seed'])
            values={k:cfg[k] for k in ('arrival_offsets','request_caps','dataset','gpus')}
            self.assertEqual(paired.setdefault(key,values),values)
            if cfg.get('token_axis'):
                self.assertEqual(36*sum(cfg['token_suffix_by_request']),cfg['count']*8000*cfg['static_layers'])

    def frontier(self):
        group=dict(members={'q0','q1'})
        channel=SimpleNamespace(by_request=dict(q0='g',q1='g'),groups=dict(g=group))
        return SubmissionFrontier(dict(length=8000,token_axis=False),lambda:channel)

    def test_remote_full_counters_do_not_release_decode(self):
        frontier=self.frontier()
        request=SimpleNamespace(request_id='q0-internal',num_computed_tokens=8000,num_prompt_tokens=8000)
        frontier.after_schedule(SimpleNamespace(num_scheduled_tokens={'q0-internal':8000}))
        self.assertTrue(frontier.defer_running(None,request,{}))
        frontier.after_schedule(SimpleNamespace(num_scheduled_tokens={'q1-internal':8000}))
        self.assertFalse(frontier.defer_running(None,request,{}))

    def test_selected_suffix_prevents_another_prefill_in_the_same_iteration(self):
        frontier=self.frontier()
        self.assertTrue(frontier.has_selected(None,{'q0-internal':8000}))
        self.assertTrue(frontier.defer_running(None,SimpleNamespace(request_id='q1-internal'),{'q0-internal':8000}))

    def test_a_decode_query_cannot_be_mistaken_for_a_suffix(self):
        frontier=self.frontier()
        with self.assertRaises(AssertionError):
            frontier.after_schedule(SimpleNamespace(num_scheduled_tokens={'q0-internal':1}))


if __name__=='__main__':
    unittest.main()
