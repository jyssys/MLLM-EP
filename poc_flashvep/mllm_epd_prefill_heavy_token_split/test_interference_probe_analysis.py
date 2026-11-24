"""Passing an unrelated-work latency check is not actual request-offload gain."""
import unittest

from .interference_probe_analysis import protection_gate,memory_records,memory_figure_records


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


class ProbeMemoryTest(unittest.TestCase):
    def fixture(self):
        def sample(free,allocated,visual=12,LLM=0,KV=0):
            parameters=dict(visual=visual)
            if LLM:parameters['language_model']=LLM
            return dict(device_total_bytes=1000,device_free_bytes=free,allocated_bytes=allocated,
                reserved_bytes=allocated+20,maximum_allocated_bytes=allocated+50,
                model_parameter_logical_bytes=sum(parameters.values()),
                model_parameter_bytes_by_prefix=parameters,unique_KV_storage_bytes=KV)
        return memory_records(0,dict(before={'18840':sample(800,30)},
            with_helper={'18840':sample(500,30),'18843':sample(490,200,LLM=100,KV=80)},
            final={'18840':sample(400,30),'18843':sample(390,200,LLM=100,KV=80)}))

    def test_shared_device_is_counted_once_per_phase(self):
        rows=memory_figure_records(self.fixture())
        actual=[r for r in rows if r['metric']=='device_used_bytes']
        self.assertEqual([r['bytes'] for r in actual],[200,500,600])
        self.assertTrue(all(r['role']=='E' and r['GPU']==4 for r in actual))

    def test_missing_helper_is_not_fabricated_zero(self):
        rows=memory_figure_records(self.fixture())
        helper=[r for r in rows if r['role']=='E_helper']
        self.assertEqual(len(helper),4)
        self.assertNotIn('before',[r['phase'] for r in helper])
        self.assertTrue(all(r['bytes']>0 for r in helper))

    def test_parameters_allocator_and_peak_scope_are_distinct(self):
        helper=next(r for r in self.fixture() if r['role']=='E_helper')
        self.assertEqual(helper['model_parameter_logical_bytes'],112)
        self.assertEqual(helper['unique_KV_storage_bytes'],80)
        self.assertEqual(helper['process_reserved_bytes'],220)
        self.assertIn('Process lifetime',helper['peak_scope'])


if __name__=='__main__':unittest.main()
