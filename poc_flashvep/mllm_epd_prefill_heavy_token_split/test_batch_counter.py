"""The observer must preserve outputs and count the prepared query positions."""
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from . import batch_counter as module


class CounterTest(unittest.TestCase):
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
