"""The counter cannot waive a noisy metric or treat periods as starts."""
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from . import batch_counter_control as control
from .batch_counter_control import qualify,METRICS
from .common import write


class CounterGateTest(unittest.TestCase):
    def fixture(self):return {stage:{key:100. for key in METRICS} for stage in ['clean_before','counter','clean_after']}

    def test_all_metrics_and_collector_are_required(self):
        value=self.fixture();self.assertTrue(qualify(value,True)['observer_gate_passed'])
        self.assertFalse(qualify(value,False)['observer_gate_passed'])
        value['counter']['ITL_p99_ms']=106.
        self.assertFalse(qualify(value,True)['observer_gate_passed'])

    def test_clean_spread_cannot_be_removed_after_observing_a_failure(self):
        value=self.fixture();value['clean_before']['TTFT_p95_ms']=96.;value['clean_after']['TTFT_p95_ms']=104.
        self.assertFalse(qualify(value,True)['observer_gate_passed'])


class CounterProcedureTest(unittest.IsolatedAsyncioTestCase):
    async def procedure(self,images):
        events=[]
        with TemporaryDirectory() as directory:
            root=Path(directory);manifest=root/'manifest.json';write(manifest,{})
            rows=[dict(request_id=str(i),image_count=1,visual_tokens=2) for i in range(100)]
            counts={r['request_id']:5 for r in rows}
            for port,role in [(18840,'encoder'),(18841,'prefill'),(18842,'decode')]:
                folder=root/'counter'/(role+'_gpu'+str(port-18836))
                histogram=[dict(output_images=images,output_visual_rows=images*2,calls=1)] if role=='encoder' else []
                write(folder/'histograms.json',dict(encoder_output_histogram=histogram))
            async def execute(actual,urls,name,qps,seed,output,**kwargs):
                events.append(('execute',name))
                self.assertEqual(actual,rows);self.assertEqual(output,counts)
                self.assertEqual(qps,2.);self.assertEqual(seed,2026112402)
                return [dict(request_id=r['request_id'],status='MEASURED_ONLINE',output_tokens=5) for r in rows]
            def rpc(port,operation,body):
                events.append((operation,port));role={18840:'encoder',18841:'prefill',18842:'decode'}[port]
                return [dict(valid=True,original_methods_restored=True,folder=str(root/'counter'/(role+'_gpu'+str(port-18836))))]
            with patch.object(control,'ROOT',root),patch.object(control,'execute',side_effect=execute),\
                    patch.object(control,'ownership_empty'),patch.object(control,'rpc',side_effect=rpc),\
                    patch.object(control,'summarize',return_value={key:100. for key in METRICS}):
                result=await control.measurements('fixture','NativeCommon','unused',manifest,rows,[],counts)
            self.assertEqual(len(result),4)
            qualification=__import__('json').loads((root/'analysis/fixture_counter_qualification.json').read_text())
        return events,qualification

    async def test_actual_procedure_keeps_full_replay_and_restores_all_workers_before_clean_after(self):
        events,result=await self.procedure(100)
        self.assertEqual(events,[('execute','fixture_warmup'),('execute','fixture_clean_before'),
            ('batch_counter/begin',18840),('batch_counter/begin',18841),('batch_counter/begin',18842),
            ('execute','fixture_counter'),('batch_counter/end',18842),('batch_counter/end',18841),
            ('batch_counter/end',18840),('execute','fixture_clean_after')])
        self.assertEqual(result['independent_initializations'],1)
        self.assertTrue(result['observer_gate_passed']);self.assertFalse(result['full_GO'])

    async def test_matching_timing_does_not_hide_missing_encoder_image_outputs(self):
        _,result=await self.procedure(99)
        self.assertFalse(result['observer_gate_passed'])
        self.assertFalse(result['encoder_output_conservation']['passed'])


if __name__=='__main__':unittest.main()
