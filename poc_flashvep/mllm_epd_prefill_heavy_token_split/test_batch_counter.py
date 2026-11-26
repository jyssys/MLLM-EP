"""The observer must preserve outputs and count the prepared query positions."""
from pathlib import Path
import json
import sys
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from . import batch_counter as module


class CounterTest(unittest.TestCase):
    def test_decode_v1_records_actual_forward_padding_and_restores_model_forward(self):
        from types import ModuleType
        context=SimpleNamespace(batch_descriptor=SimpleNamespace(num_reqs=2,num_tokens=8),
            cudagraph_runtime_mode=SimpleNamespace(name='FULL'))
        context_module=ModuleType('vllm.forward_context')
        context_module.get_forward_context=lambda:context
        sentinel=object()

        class Runner:
            input_batch=SimpleNamespace(req_ids=['d'],num_reqs=1,
                num_computed_tokens_cpu=np.array([12]),num_prompt_tokens=np.array([12]))
            def _prepare_inputs(self,output,scheduled):return None,None
            def _model_forward(self,**kwargs):return sentinel
            def execute_model(self,output):
                self._prepare_inputs(output,np.array([1]))
                return self._model_forward(input_ids=np.zeros(context.batch_descriptor.num_tokens))

        output=SimpleNamespace(num_scheduled_tokens={'d':1},total_num_scheduled_tokens=1,
            scheduled_encoder_inputs={},scheduled_spec_decode_tokens={},
            scheduled_cached_reqs=SimpleNamespace(req_ids=['d'],num_output_tokens=[1]))
        with TemporaryDirectory() as directory,patch.object(module,'ROOT',Path(directory)),\
                patch.dict(sys.modules,{'vllm.forward_context':context_module}),\
                patch.dict(module.os.environ,MLLM_EPD_ROLE='decode',CUDA_VISIBLE_DEVICES='6',
                           MLLM_EPD_DECODE_COUNTER_IDENTITIES='1'):
            runner=Runner();counter=module.BatchCounter(runner,'v1_decode_fixture',decode_only=True)
            info=counter.begin()
            self.assertIs(runner.execute_model(output),sentinel)
            context.batch_descriptor.num_reqs=None
            context.batch_descriptor.num_tokens=1
            context.cudagraph_runtime_mode.name='NONE'
            self.assertIs(runner.execute_model(output),sentinel)
            result=counter.close()
            values=json.loads((Path(result['folder'])/'histograms.json').read_text())
            identities=[json.loads(line) for line in (Path(result['folder'])/'decode_identities.jsonl').read_text().splitlines()]
        self.assertEqual(info['runner_API'],'V1_PRIVATE_PREPARE_AND_FORWARD_CONTEXT')
        self.assertTrue(result['valid'])
        self.assertEqual(result['decode_maxima']['prepared_decode_requests'],1)
        self.assertEqual(result['decode_maxima']['prepared_requests_after_padding'],2)
        self.assertEqual(values['decode_identity_steps'],2)
        self.assertTrue(all(step['end']>=step['begin'] and step['requests']==[
            dict(request_id='d',prompt_rows=0,decode_rows=1)] for step in identities))
        self.assertEqual(values['prepared_decode_padding_histogram'],
            [dict(requests=1,decode_requests=1,query_rows=1,requests_after_padding=1,
                query_rows_after_padding=1,graph_mode='NONE',calls=1),
             dict(requests=1,decode_requests=1,query_rows=1,requests_after_padding=2,
                query_rows_after_padding=8,graph_mode='FULL',calls=1)])
        self.assertNotIn('_model_forward',vars(runner))

    def test_decode_v2_records_logical_rows_and_real_padding_without_encoder(self):
        prepared=SimpleNamespace(req_ids=['a','b'],num_reqs=2,
            num_scheduled_tokens=np.array([3,1]),num_tokens=4,
            num_computed_prefill_tokens_np=np.array([2,6]),prefill_len_np=np.array([4,6]),
            num_draft_tokens=0,num_reqs_after_padding=4,num_tokens_after_padding=8)

        class Runner:
            def prepare_inputs(self,output,descriptor):return prepared
            def execute_model(self,output):
                return self.prepare_inputs(output,SimpleNamespace(cg_mode=SimpleNamespace(name='FULL')))

        output=SimpleNamespace(num_scheduled_tokens={'a':3,'b':1},total_num_scheduled_tokens=4,
            scheduled_encoder_inputs={},
            scheduled_cached_reqs=SimpleNamespace(req_ids=['a','b'],num_output_tokens=[0,7]))
        with TemporaryDirectory() as directory,patch.object(module,'ROOT',Path(directory)),\
                patch.dict(module.os.environ,MLLM_EPD_ROLE='decode',CUDA_VISIBLE_DEVICES='6'):
            runner=Runner();counter=module.BatchCounter(runner,'decode_fixture',decode_only=True)
            counter.begin()
            self.assertIs(runner.execute_model(output),prepared)
            result=counter.close()
            values=json.loads((Path(result['folder'])/'histograms.json').read_text())
        self.assertTrue(result['valid'])
        self.assertEqual(values['scheduler_generation_phase_histogram'],
            [dict(requests=2,generation_phase_requests=1,query_rows=4,calls=1)])
        self.assertEqual(values['prepared_decode_padding_histogram'],
            [dict(requests=2,decode_requests=2,query_rows=4,requests_after_padding=4,
                query_rows_after_padding=8,graph_mode='FULL',calls=1)])
        self.assertFalse(values['encoder_output_histogram'])
        self.assertNotIn('prepare_inputs',vars(runner))

    def test_v2_real_call_order_counts_returned_batch_and_restores_nested_encoder(self):
        sentinel=object()
        batch=SimpleNamespace(num_reqs=2,num_scheduled_tokens=np.array([2,1]),num_tokens=3,
            num_computed_prefill_tokens_np=np.array([6,8]),prefill_len_np=np.array([8,8]),num_draft_tokens=0)
        class Encoder:
            def execute_mm_encoder(self,kwargs):return [np.empty((10,4)),np.empty((20,4))]
        class Runner:
            def __init__(self):self.model_state=SimpleNamespace(encoder_runner=Encoder())
            def prepare_inputs(self,output,descriptor):return batch
            def execute_model(self,output):
                self.prepare_inputs(output,object())
                self.model_state.encoder_runner.execute_mm_encoder([])
                return sentinel
        runner=Runner();encoder=runner.model_state.encoder_runner
        output=SimpleNamespace(num_scheduled_tokens={'p':2,'d':1},total_num_scheduled_tokens=3,scheduled_encoder_inputs={'p':[0,1]})
        with TemporaryDirectory() as directory,patch.object(module,'ROOT',Path(directory)),\
                patch.dict(module.os.environ,MLLM_EPD_ROLE='agg',CUDA_VISIBLE_DEVICES='4'):
            counter=module.BatchCounter(runner,'v2_fixture');counter.begin()
            self.assertIs(runner.execute_model(output),sentinel)
            result=counter.close()
        self.assertTrue(result['valid']);self.assertEqual(counter.prepared[(2,1,1,0,3,2,1)],1)
        self.assertEqual(counter.encoded[(2,30)],1)
        self.assertNotIn('prepare_inputs',vars(runner));self.assertNotIn('execute_mm_encoder',vars(encoder))

    def test_v2_no_draft_or_geometry_error_may_be_silently_named_valid(self):
        batch=SimpleNamespace(num_reqs=1,num_scheduled_tokens=np.array([2]),num_tokens=2,
            num_computed_prefill_tokens_np=np.array([6]),prefill_len_np=np.array([8]),num_draft_tokens=1)
        with self.assertRaises(AssertionError):module.v2_query_geometry(batch)

    def test_last_prompt_chunk_is_not_reclassified_as_decode(self):
        batch=SimpleNamespace(num_reqs=1,num_computed_tokens_cpu=np.array([8190]),num_prompt_tokens=np.array([8192]))
        self.assertEqual(module.query_geometry(batch,np.array([2])),(1,1,0,0,2,2,0))

    def test_mixed_rows_and_request_categories_conserve_actual_queries(self):
        batch=SimpleNamespace(num_reqs=3,num_computed_tokens_cpu=np.array([0,100,9]),num_prompt_tokens=np.array([8,100,10]))
        self.assertEqual(module.query_geometry(batch,np.array([4,1,3])),(3,1,1,1,8,5,3))

    def fixture(self):
        sentinel=object()
        class Runner:
            def __init__(self):
                self.input_batch=SimpleNamespace(num_reqs=1,num_computed_tokens_cpu=np.array([0]),num_prompt_tokens=np.array([8]))
            def execute_model(self,output):
                self._prepare_inputs(output,np.array([4]));self._execute_mm_encoder(output)
                return sentinel
            def _prepare_inputs(self,output,scheduled):return sentinel
            def _execute_mm_encoder(self,output):return [np.empty((10,4)),np.empty((20,4))]
        return Runner(),sentinel

    def test_wrapped_real_methods_keep_output_identity_and_restore_class_slots(self):
        runner,sentinel=self.fixture();original=runner.execute_model.__func__
        output=SimpleNamespace(num_scheduled_tokens={'r':4},total_num_scheduled_tokens=4,scheduled_encoder_inputs={'r':[0,1]})
        with TemporaryDirectory() as directory,patch.object(module,'ROOT',Path(directory)),\
                patch.dict(module.os.environ,MLLM_EPD_ROLE='prefill',CUDA_VISIBLE_DEVICES='5'):
            counter=module.BatchCounter(runner,'fixture');counter.begin()
            self.assertIs(runner.execute_model(output),sentinel)
            result=counter.close()
        self.assertTrue(result['valid']);self.assertEqual(result['prepared_calls'],1)
        self.assertEqual(counter.prepared[(1,1,0,0,4,4,0)],1)
        self.assertEqual(counter.encoded[(2,30)],1)
        self.assertEqual(runner.execute_model.__func__,original)
        self.assertNotIn('execute_model',vars(runner))

    def test_unavailable_observation_does_not_modify_or_block_the_model_result(self):
        runner,sentinel=self.fixture();runner.input_batch.num_reqs=2
        output=SimpleNamespace(num_scheduled_tokens={'r':4},total_num_scheduled_tokens=4,scheduled_encoder_inputs={})
        with TemporaryDirectory() as directory,patch.object(module,'ROOT',Path(directory)),\
                patch.dict(module.os.environ,MLLM_EPD_ROLE='prefill',CUDA_VISIBLE_DEVICES='5'):
            counter=module.BatchCounter(runner,'fixture');counter.begin()
            self.assertIs(runner.execute_model(output),sentinel)
            result=counter.close()
        self.assertFalse(result['valid']);self.assertTrue(result['errors'])


if __name__=='__main__':unittest.main()
