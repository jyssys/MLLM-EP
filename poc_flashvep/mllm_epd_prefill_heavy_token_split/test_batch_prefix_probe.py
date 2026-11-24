"""CPU fixtures exercise the actual post-forward wrapper and per-row offsets."""
import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch

from . import batch_prefix_probe as probe


class BatchPrefixProbeTest(unittest.TestCase):
    def test_runtime_id_validation(self):
        self.assertEqual(probe.request_base('chatcmpl-a-b:P-deadbeef','P',{'a-b':{}}),'a-b')
        for rid in ['chatcmpl-a-b:D-deadbeef','chatcmpl-other:P-deadbeef','a-b:P-deadbeef']:
            with self.assertRaises(AssertionError):probe.request_base(rid,'P',{'a-b':{}})

    def test_actual_wrappers_update_batch_then_capture_both_rows(self):
        contracts={'first':dict(prefix_tokens=2,full_prompt_tokens=4),
                   'second':dict(prefix_tokens=3,full_prompt_tokens=6)}
        ids=['chatcmpl-first:P-deadbeef','chatcmpl-second:P-deadbeef']
        mrope=[torch.arange(4).repeat(3,1),torch.arange(6).repeat(3,1)+10]
        positions=torch.cat([mrope[0][:,:2],mrope[1][:,:3]],dim=1)
        caches=[torch.arange(2*4*128*256,dtype=torch.float32).reshape(2,4,128,256)+i for i in range(28)]
        table=SimpleNamespace(num_blocks_per_row=[1,1],block_table=SimpleNamespace(np=np.array([[0],[1]])),block_size=128)
        batch=SimpleNamespace(req_ids=ids,num_prompt_tokens=[2,3],num_computed_tokens_cpu=[0,0],block_table=[table])
        states={rid:SimpleNamespace(prompt_token_ids=list(range(len(pos[0]))),mrope_positions=pos,mrope_position_delta=0)
                for rid,pos in zip(ids,mrope)}
        result=object()
        class Runner:
            def __init__(self):
                self.kv_caches=caches;self.requests=states;self.input_batch=None;self.forward_calls=0
            def _model_forward(self,**kwargs):
                self.forward_calls+=1
                self.assert_positions=kwargs['positions']
                return result
            def execute_model(self,output):
                self.input_batch=batch
                return self._model_forward(positions=positions)
        runner=Runner();output=SimpleNamespace(num_scheduled_tokens=dict(zip(ids,[2,3])),total_num_scheduled_tokens=5)
        with TemporaryDirectory() as directory,patch.object(probe,'ROOT',Path(directory)),\
             patch.dict(os.environ,MLLM_EPD_ROLE='prefill'),patch.object(probe,'forward_context',return_value=SimpleNamespace(
                cudagraph_runtime_mode='NONE',batch_descriptor='actual_cpu_fixture')):
            obj=probe.BatchPrefixProbe(SimpleNamespace(model_runner=runner),'mixed',contracts)
            obj.begin()
            self.assertIs(runner.execute_model(output),result)
            self.assertEqual(runner.forward_calls,1)
            self.assertIs(runner.assert_positions,positions)
            obj.close()
            record=json.loads((Path(directory)/'raw/batch_prefix/mixed/manifest.json').read_text())
            self.assertEqual(set(record['snapshots']),set(contracts))
            self.assertTrue(all(len(row['layers'])==28 for row in record['snapshots'].values()))
            self.assertEqual([r['query_offset'] for r in record['events'][0]['requests']],[0,2])
            self.assertEqual(record['events'][0]['actual_batch'],2)
            self.assertNotIn('execute_model',vars(runner))
            self.assertNotIn('_model_forward',vars(runner))


if __name__=='__main__':unittest.main()
