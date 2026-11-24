"""Passing an unrelated-work latency check is not actual request-offload gain."""
import unittest

from .interference_probe_analysis import protection_gate


class ProbeProtectionTest(unittest.TestCase):
    def fixture(self):
        pairs=[dict(arm='E128',mode='replay',reference=reference,metric=metric,
            paired_start_bootstrap95_CI=[.98,1.02]) for reference in
            ['NativeStrong','NativeCommon','ResidentNoProbeMean'] for metric in
            ['E_HTTP_p95_ms','ITL_p99_ms','TPOT_p95_ms','TTFT_p50_ms','TTFT_p95_ms']]
        starts=[dict(arm='E128',mode='replay',block=index,decode_pass_percent=98,accepted_probes=20) for index in range(3)]
        return pairs,starts

    def test_probe_pass_is_not_encoder_exclusive_or_serving_gain(self):
        pairs,starts=self.fixture();value=protection_gate(pairs,starts,'E128','replay')
        self.assertTrue(value['observed_HTTP_protection_gate'])
        self.assertFalse(value['serving_speedup']);self.assertFalse(value['real_request_prefill_offload'])
        self.assertEqual(value['encoder_exclusive_GPU_service_gate'],'NOT_MEASURED')

    def test_zero_added_work_or_decode_failure_cannot_pass(self):
        pairs,starts=self.fixture();starts[1]['accepted_probes']=0
        self.assertFalse(protection_gate(pairs,starts,'E128','replay')['observed_HTTP_protection_gate'])
        pairs,starts=self.fixture();starts[1]['decode_pass_percent']=94
        self.assertFalse(protection_gate(pairs,starts,'E128','replay')['absolute_decode_all_starts'])

    def test_regression_against_either_native_is_retained(self):
        pairs,starts=self.fixture();pairs[0]['paired_start_bootstrap95_CI'][1]=1.051
        self.assertFalse(protection_gate(pairs,starts,'E128','replay')['observed_HTTP_protection_gate'])


if __name__=='__main__':unittest.main()
