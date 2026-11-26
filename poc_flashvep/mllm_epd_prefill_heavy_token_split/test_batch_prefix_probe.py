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
    def test_decode_positions_and_full_prompt_capture_are_explicit_d_only(self):
        contracts={'r':dict(prefix_tokens=2,full_prompt_tokens=4)}
        rid='chatcmpl-r:D-deadbeef';positions=torch.arange(4).repeat(3,1)
        cache=torch.ones(1,4,128,256)
        table=SimpleNamespace(num_blocks_per_row=[1],block_table=SimpleNamespace(np=np.array([[0]])),block_size=128)
        batch=SimpleNamespace(req_ids=[rid],num_prompt_tokens=[4],num_computed_tokens_cpu=[2],block_table=[table])
        state=SimpleNamespace(prompt_token_ids=list(range(4)),mrope_positions=positions,mrope_position_delta=-1)
        class Runner:
            def __init__(self):
                self.kv_caches=[cache]*28;self.requests={rid:state};self.input_batch=batch
            def _model_forward(self,**kwargs):return 'original'
            def execute_model(self,output):
                start=self.input_batch.num_computed_tokens_cpu[0]
                actual=positions[...,2:] if start==2 else torch.tensor([[3],[3],[3]])
                return self._model_forward(positions=actual)
        with TemporaryDirectory() as directory,patch.object(probe,'ROOT',Path(directory)),\
             patch.dict(os.environ,MLLM_EPD_ROLE='decode'),patch.object(probe,'forward_context',return_value=SimpleNamespace(
                cudagraph_runtime_mode='NONE',batch_descriptor='cpu_fixture')):
            runner=Runner();obj=probe.BatchPrefixProbe(SimpleNamespace(model_runner=runner),'natural',contracts,True,True)
            obj.begin()
            self.assertEqual(runner.execute_model(SimpleNamespace(num_scheduled_tokens={rid:2},total_num_scheduled_tokens=2)),'original')
            batch.num_computed_tokens_cpu[0]=4
            self.assertEqual(runner.execute_model(SimpleNamespace(num_scheduled_tokens={rid:1},total_num_scheduled_tokens=1)),'original')
            obj.close()
            record=json.loads((Path(directory)/'raw/batch_prefix/natural/manifest.json').read_text())
            self.assertEqual([e['requests'][0]['phase'] for e in record['events']],['prefill','decode'])
            self.assertEqual(set(record['full_prompt_tensor_files']),{'r'})
            self.assertTrue(Path(record['full_prompt_tensor_files']['r']['path']).exists())

    def test_runtime_id_validation(self):
        self.assertEqual(probe.request_base('chatcmpl-a-b:P-deadbeef','P',{'a-b':{}}),'a-b')
        for rid in ['chatcmpl-a-b:D-deadbeef','chatcmpl-other:P-deadbeef','a-b:P-deadbeef']:
            with self.assertRaises(AssertionError):probe.request_base(rid,'P',{'a-b':{}})

    def test_eager_inplace_residual_preserves_original_input_receipt(self):
        rid = 'chatcmpl-r:P-deadbeef'
        positions = torch.arange(4).repeat(3, 1)
        embeddings = torch.ones((1, 3), dtype=torch.bfloat16)
        expected = probe.tensor_signature(embeddings)
        contracts = {'r': dict(prefix_tokens=2, full_prompt_tokens=4, prefill_full_prompt=True)}

        class Runner:
            model_config = SimpleNamespace(enforce_eager=True)
            kv_caches = [None] * 28
            input_batch = SimpleNamespace(req_ids=[rid], num_prompt_tokens=[4], num_computed_tokens_cpu=[0])
            requests = {rid: SimpleNamespace(mrope_positions=positions)}

            def execute_model(self, output):
                return self._model_forward(positions=positions[:, :1], inputs_embeds=embeddings)

            def _model_forward(self, positions, inputs_embeds):
                inputs_embeds.add_(1)

        with TemporaryDirectory() as directory, patch.object(probe, 'ROOT', Path(directory)), \
             patch.dict(os.environ, MLLM_EPD_ROLE='prefill'), patch.object(
                 probe, 'forward_context', return_value=SimpleNamespace(
                     cudagraph_runtime_mode='NONE', batch_descriptor='eager_fixture'
                 )
             ):
            runner = Runner()
            capture = probe.BatchPrefixProbe(SimpleNamespace(model_runner=runner), 'eager', contracts)
            capture.begin()
            runner.execute_model(SimpleNamespace(num_scheduled_tokens={rid: 1}, total_num_scheduled_tokens=1))
            capture.close()
            record = json.loads((Path(directory) / 'raw/batch_prefix/eager/manifest.json').read_text())
        actual = record['events'][0]['requests'][0]
        self.assertEqual(actual['inputs_embeds'], expected)
        self.assertTrue(actual['inputs_embeds_changed_during_forward'])
        self.assertNotEqual(probe.tensor_signature(embeddings), expected)

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
