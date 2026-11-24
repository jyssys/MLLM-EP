"""Keep nonexistent P stages and all Native-relative protection checks explicit."""
import unittest

from .whole_deflection_analysis import host_path,COMMON_STAGES,decision


class WholeAnalysisTest(unittest.TestCase):
    def fixture(self):
        request=dict(scheduled_arrival=0.,token_times=[10.],TTFT_ms=10000.)
        trace=dict(received=1.,E_submit=2.,E_complete=4.,D_submit=7.,deflected=True,
            P_stage='NOT_APPLICABLE: direct E to D',KV_handoff='NONE: no prefix KV',
            D_admission=dict(begin=5.,admitted=6.,extra_prefill=True,before=dict(active_http_requests=0)))
        return request,trace

    def test_deflected_p_duration_is_absent_not_zero(self):
        request,trace=self.fixture();value=host_path(request,trace)
        self.assertIsNone(value['P_HTTP_enclosure_ms'])
        self.assertEqual(value['P_stage'],'NOT_APPLICABLE')
        self.assertEqual(sum(value[k] for k in COMMON_STAGES),request['TTFT_ms'])
        self.assertEqual(value['D_coordinator_wait_ms'],1000.)

    def test_native_without_a_coordinator_gate_does_not_receive_a_fake_gate_time(self):
        request,trace=self.fixture();trace.update(deflected=False,P_submit=5.,P_ready=6.)
        trace.pop('D_admission');value=host_path(request,trace)
        self.assertEqual(value['P_HTTP_enclosure_ms'],1000.)
        self.assertIsNone(value['D_coordinator_wait_ms'])

    def gates(self):
        values={'TTFT_p50_ms':(.98,[.96,1.01]),'TTFT_p95_ms':(.90,[.88,.94]),
            'ITL_p99_ms':(1.,[.98,1.02]),'TPOT_p95_ms':(1.,[.98,1.02]),
            'SLO_goodput':(1.12,[1.08,1.16]),'raw_output_tokens_per_second':(1.02,[1.0,1.04]),
            'joint_percent':(1.01,[1.,1.02])}
        pairs=[dict(subset='Main100',mode='replay',reference=reference,metric=key,
            ratio_of_start_means=point,paired_start_bootstrap95_CI=ci.copy())
            for reference in ['NativeStrong','NativeOnDemand'] for key,(point,ci) in values.items()]
        starts=[dict(subset='Main100',mode='replay',method='WholeDeflection',block=i,decode_pass_percent=99) for i in range(3)]
        return pairs,starts

    def test_runtime_threshold_pass_is_not_full_quality_or_generic_token_go(self):
        pairs,starts=self.gates();value=decision(pairs,starts,'Main100','replay')
        self.assertTrue(value['relative_decode_gate']);self.assertFalse(value['full_GO'])
        self.assertEqual(value['full_task_quality'],'PENDING')
        self.assertIn('not a Generic Token Split',value['scope'])

    def test_failure_against_one_native_or_one_start_cannot_be_hidden(self):
        pairs,starts=self.gates();pairs[2]['paired_start_bootstrap95_CI'][1]=1.051
        self.assertEqual(decision(pairs,starts,'Main100','replay')['runtime_decision'],'NO_GO_ON_TESTED_RUNTIME')
        pairs,starts=self.gates();starts[0]['decode_pass_percent']=94
        self.assertFalse(decision(pairs,starts,'Main100','replay')['absolute_decode_all_starts'])


if __name__=='__main__':unittest.main()
