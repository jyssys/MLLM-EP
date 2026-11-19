import unittest
from .cohort_axis_oracle import beam,rule,token_deltas
from .replay import Cost


class CohortOracleTests(unittest.TestCase):
    def table(self):
        return {'Layer':Cost(.3,.05,.35),'Token':Cost(.27,.10,.35)}

    def test_beam_keeps_both_fixed_controls_and_exact_work(self):
        result=beam([i*.1 for i in range(4)],self.table(),8,width=8,deltas=[-1,1,-1,1])
        self.assertEqual(result['final_row_delta'],0)
        self.assertGreaterEqual(result['oracle']['qualified_output_tokens_s'],
            max(r['qualified_output_tokens_s'] for r in result['fixed'].values()))

    def test_actual_Decode4_plan_has_exact_aggregate_rows(self):
        deltas=token_deltas(8,72,4)
        self.assertEqual(len(deltas),18)
        self.assertEqual(sum(deltas),0)

    def test_shadow_rule_closes_work_without_reading_future_finishes(self):
        for kind in ('one','two'):
            result=rule([i*.1 for i in range(4)],self.table(),8,4.,kind=kind,deltas=[-1,1,-1,1])
            self.assertEqual(result['final_row_delta'],0)
            self.assertEqual(len(result['decisions']),4)

    def test_constant_actions_need_no_adaptive_headroom(self):
        table={'Layer':Cost(.2,.1,.3),'Token':Cost(.2,.1,.3)}
        result=beam([i*.1 for i in range(4)],table,8,width=8,deltas=[-1,1,-1,1])
        self.assertEqual(result['oracle']['TTFT_p95_ms'],result['fixed']['Layer']['TTFT_p95_ms'])


if __name__=='__main__':
    unittest.main()
