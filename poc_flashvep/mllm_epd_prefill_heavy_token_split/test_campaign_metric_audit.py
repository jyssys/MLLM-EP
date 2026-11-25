"""Use actual timestamp and SLO boundaries, including empty single-token ITL."""
import unittest

from .campaign_metric_audit import request_contract,non_serving_reason


class MetricAuditTest(unittest.TestCase):
    def test_known_correctness_runs_are_not_missing_serving_cohorts(self):
        for run in ['base_C14_agg_S1_N1_natural_v0','ec_NATIVE_C11_S2_N8_natural_v0',
                'token_CONFIRM_C37_B0_Static10_KV_TEACHER','token_CONFIRM_C37_B0_Static10_KV_TOKEN']:
            self.assertEqual(non_serving_reason(run),'CORRECTNESS_ONLY')

    def test_real_incomplete_or_unknown_run_is_not_silently_excluded(self):
        for run in ['token_CONFIRM_C37_B0_Static10_Main100_natural',
                'batch_COUNTER_C46_B0_AggDP3_counter','base_C14_agg_S1_unrecognized_failure']:
            self.assertIsNone(non_serving_reason(run))
        self.assertEqual(non_serving_reason('batch_COUNTER_C46_B0_AggDP3_warmup'),'WARMUP_OR_SMOKE')

    def sample(self,times):
        count=len(times);finish=times[-1]+.1
        return dict(status='MEASURED_ONLINE',scheduled_arrival=0.,token_times=times,token_ids=list(range(count)),
            output_tokens=count,usage=dict(completion_tokens=count),finish=finish,TTFT_ms=times[0]*1000,
            TPOT_ms=(times[-1]-times[0])*1000/(count-1) if count>1 else 0.,
            ITL_ms=[(b-a)*1000 for a,b in zip(times,times[1:])],E2E_ms=finish*1000)

    def test_single_output_empty_intervals_and_zero_tpot_are_valid(self):
        self.assertTrue(request_contract(self.sample([5.]))['JOINT_PASS'])

    def test_threshold_equality_is_not_a_strict_less_than_gate(self):
        row=self.sample([0.,.033]);self.assertTrue(request_contract(row)['JOINT_PASS'])
        row=self.sample([5.000001]);self.assertEqual(request_contract(row)['category'],'TTFT_FAIL')

    def test_tail_violation_survives_good_tpot(self):
        row=self.sample([1.,1.,1.04,1.04])
        self.assertEqual(request_contract(row)['category'],'ITL_FAIL')

    def test_invalid_recorded_interval_cannot_pass_from_cached_metrics(self):
        row=self.sample([1.,1.01]);row['ITL_ms']=[1.]
        with self.assertRaises(AssertionError):request_contract(row)


if __name__=='__main__':unittest.main()
