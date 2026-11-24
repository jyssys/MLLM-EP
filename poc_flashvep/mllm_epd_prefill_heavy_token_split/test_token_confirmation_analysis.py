"""Paired starts, not request or token observations, set uncertainty."""
import unittest


class PairedIntervalTest(unittest.TestCase):
    def test_constant_paired_effect_is_exact_under_varying_start_costs(self):
        from .token_confirmation_analysis import paired_interval
        result=paired_interval([9.,18.,90.],[10.,20.,100.])
        self.assertAlmostEqual(result['ratio_of_start_means'],.9)
        self.assertTrue(all(abs(v-.9)<1e-12 for v in result['paired_start_bootstrap95_CI']))
        self.assertEqual(result['independent_start_pairs'],3)
        self.assertEqual(result['empirical_resamples'],27)

    def test_reject_request_count_as_independent_start_count(self):
        from .token_confirmation_analysis import paired_interval
        with self.assertRaises(AssertionError):paired_interval([1.]*100,[1.]*100)

    def fixture(self):
        common=dict(method='Idle10',subset='Main100',mode='replay')
        values={'TTFT_p95_ms':(.90,[.88,.94]),'SLO_goodput':(1.10,[1.07,1.13]),
            'ITL_p99_ms':(1.,[.99,1.02]),'TPOT_p95_ms':(1.,[.99,1.02]),'TTFT_p50_ms':(.99,[.97,1.01])}
        pairs=[dict(**common,baseline=baseline,metric=key,ratio_of_start_means=point,paired_start_bootstrap95_CI=ci.copy())
            for baseline in ['NativeStrong','NativeCommon'] for key,(point,ci) in values.items()]
        starts=[dict(**common,block=i,ITL_pass_percent=96.,TPOT_pass_percent=100.) for i in range(3)]
        return pairs,starts

    def test_runtime_success_cannot_be_promoted_without_task_quality(self):
        from .token_confirmation_analysis import runtime_decision
        pairs,starts=self.fixture()
        result=runtime_decision(pairs,starts,'Idle10','Main100','replay')
        self.assertEqual(result['runtime_decision'],'RUNTIME_CONDITIONAL_QUALITY_PENDING')
        self.assertFalse(result['full_GO'])

    def test_failure_against_either_native_or_absolute_decode_is_retained(self):
        from .token_confirmation_analysis import runtime_decision
        pairs,starts=self.fixture()
        row=next(r for r in pairs if r['baseline']=='NativeStrong' and r['metric']=='ITL_p99_ms')
        row['paired_start_bootstrap95_CI'][1]=1.051
        result=runtime_decision(pairs,starts,'Idle10','Main100','replay')
        self.assertEqual(result['runtime_decision'],'NO_GO_ON_TESTED_RUNTIME')
        self.assertFalse(result['decode_relative_gate'])
        pairs,starts=self.fixture();starts[2]['ITL_pass_percent']=94.
        result=runtime_decision(pairs,starts,'Idle10','Main100','replay')
        self.assertFalse(result['absolute_decode_all_starts'])
        self.assertEqual(result['runtime_decision'],'NO_GO_ON_TESTED_RUNTIME')


if __name__=='__main__':unittest.main()
