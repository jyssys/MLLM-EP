import unittest
from .axis_oracle import exhaustive,pressure_rule,demand_scores,demand_rule,cycle_boundary
from .replay import Cost


class OracleTests(unittest.TestCase):
    def test_equal_actions_have_no_oracle_gain(self):
        table={'Layer':Cost(.2,.1,.3),'Token':Cost(.2,.1,.3)}
        result=exhaustive([i*.1 for i in range(36)],table)
        self.assertEqual(result['tested_sequences'],4)
        self.assertEqual(result['oracle']['TTFT_p95_ms'],result['fixed']['Layer']['TTFT_p95_ms'])

    def test_oracle_contains_both_fixed_sequences(self):
        table={'Layer':Cost(.3,.05,.3),'Token':Cost(.2,.15,.3)}
        result=exhaustive([i*.1 for i in range(36)],table)
        self.assertGreaterEqual(result['oracle']['qualified_output_tokens_s'],
            max(r['qualified_output_tokens_s'] for r in result['fixed'].values()))

    def test_rule_commits_only_at_matched_work_boundary(self):
        table={'Layer':Cost(.3,.05,.3),'Token':Cost(.2,.15,.3)}
        result=pressure_rule([i*.1 for i in range(36)],table,batch=2)
        self.assertEqual([r['request_index'] for r in result['decisions']],[0,18])
        self.assertEqual(len(result['sequence']),2)

    def test_invalid_block_geometry_rejected(self):
        with self.assertRaises(AssertionError):
            exhaustive([0.]*32,{'Layer':Cost(1.,1.,1.),'Token':Cost(1.,1.,1.)})

    def test_Decode_demand_can_reverse_saturated_axis(self):
        table={'Layer':Cost(.3,.05,.35),'Token':Cost(.27,.10,.35)}
        one=demand_scores(table,1,100)['candidates']
        two=demand_scores(table,2,100)['candidates']
        self.assertEqual(min(one,key=lambda a:one[a]['score']),'Layer')
        self.assertEqual(min(two,key=lambda a:two[a]['score']),'Token')

    def test_backlog_separates_unqueued_and_cycle_objectives(self):
        table={'Layer':Cost(.3,.05,.35),'Token':Cost(.27,.10,.35)}
        low=demand_scores(table,2,1)['candidates']
        high=demand_scores(table,2,100)['candidates']
        self.assertEqual(min(low,key=lambda a:low[a]['score']),'Layer')
        self.assertEqual(min(high,key=lambda a:high[a]['score']),'Token')

    def test_Decode4_keeps_exact_block_geometry(self):
        table={'Layer':Cost(.3,.05,.35),'Token':Cost(.27,.10,.35)}
        result=demand_rule([i*.1 for i in range(72)],table,batch=4)
        self.assertEqual(result['block'],36)
        self.assertEqual(len(result['sequence']),2)

    def test_model_boundary_is_a_cycle_crossing_not_a_TTFT_claim(self):
        table={'Layer':Cost(.3,.05,.35),'Token':Cost(.27,.10,.35)}
        boundary=cycle_boundary(table,2)
        self.assertAlmostEqual(boundary['common_Decode_demand_threshold_seconds'],.2)
        for d,expected in ((.15,'Token'),(.25,'Layer')):
            values={a:max(c.P,c.preparation+d) for a,c in table.items()}
            self.assertEqual(min(values,key=values.get),expected)

    def test_dominance_needs_no_positive_threshold(self):
        table={'Layer':Cost(.2,.05,.35),'Token':Cost(.27,.10,.35)}
        self.assertEqual(cycle_boundary(table,2)['dominant'],'Layer')


if __name__=='__main__':
    unittest.main()
