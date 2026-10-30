"""CPU tests for legal token-prefix and loop-cut boundaries."""
import unittest
from .prepare import base
from .common import ROOT,MODEL_PINS

class Contracts(unittest.TestCase):
    def test_matched_huginn_candidate_fractions(self):
        from .policy import token_candidates
        self.assertEqual(token_candidates(1536,256,32,[0,8,16,24,28,32]),[0,256,768,1024,1280,1536])
    def test_token_fallback_requeues_suffix_prelude(self):
        from collections import defaultdict,deque
        from types import SimpleNamespace as S
        from .connector import PartitionConnector
        from vllm_rlt.request import RequestPhase
        from .common import config
        c=PartitionConnector.__new__(PartitionConnector);c.cfg=dict(axis='token')
        r=S(request_id='r',prompt_token_ids=list(range(512)),phase=RequestPhase.PREFILLING_RECURRENT,ltp_prefix=256,sampling_params=S(seed=1))
        t=S(tid='t',request=r,prefix=256,cut=4,awaiting_decision=True)
        queues=defaultdict(deque);queues[r.phase].append('r')
        def enqueue(request,phase):request.phase=phase;queues[phase].append(request.request_id)
        c.engine=S(scheduler=S(queues=queues,enqueue=enqueue));c.cache=S(max_loops=4);c.transfers={'t':t}
        import os
        from unittest.mock import patch
        with patch.dict(os.environ,ARC_RUN='cpu_regression'),patch('poc_flashvep.adaptive_loop_token_pd_2gpu.connector.append'):
            c.command(dict(kind='frontier_decision',tid='t',accept=False,reason='test'))
        self.assertEqual(r.phase,RequestPhase.PREFILLING_PRELUDE)
        self.assertEqual(list(queues[RequestPhase.PREFILLING_RECURRENT]),[])
        self.assertEqual(list(queues[RequestPhase.PREFILLING_PRELUDE]),['r'])
    def test_full_prefix_initializes_once(self):
        from types import SimpleNamespace as S
        import torch
        from .connector import PartitionConnector
        c=PartitionConnector.__new__(PartitionConnector)
        c.cfg=dict(axis='token',verify_transfer=False)
        request=S(request_id='r',prompt_token_ids=[1,2])
        c.transfers={'r':S(prefix=2)}
        allocation=S(written=[[S(prefix=0)]])
        c.cache=S(_get_allocation=lambda rid:allocation)
        c.engine=S(model=S(config=S(hidden_size=3)))
        c.layout=lambda t:[(0,0,2)]
        def original(*args,**kwargs):
            self.assertEqual(allocation.written[0][0].prefix,0)
            allocation.written[0][0].prefix=2
            return True
        c.original_accept=original
        self.assertTrue(c.accept_cut(request,torch.zeros(3)))
    def test_token_prefix_closes_prelude_window(self):
        from types import SimpleNamespace
        from ..adaptive_recurrence_cut.staged import install
        from .connector import install_token_barrier
        from vllm_rlt.core.scheduler import Scheduler
        install(dict(state_capacity_tokens=4096));install_token_barrier()
        request=SimpleNamespace(num_preluded_tokens=128,prompt_token_ids=[0]*256,ltp_prefix=128)
        self.assertFalse(Scheduler._prelude_window_open(None,request))
    def test_pins(self):
        self.assertEqual(MODEL_PINS['ouro'][2],4);self.assertEqual(MODEL_PINS['huginn'][2],32)
    def test_disjoint_prompts(self):
        import json
        for workload in ('W1','W2','W3','W4'):
            a=json.loads((ROOT/'data'/f'{workload}_calibration.json').read_text())
            b=json.loads((ROOT/'data'/f'{workload}_test.json').read_text())
            self.assertFalse({r['prompt_sha256'] for r in a}&{r['prompt_sha256'] for r in b})
            for row in a+b:
                self.assertTrue(all(len(v)+row['cap']<=4096 for v in row['input_ids'].values()))
    def test_prefix_candidates(self):
        for n in (128,512,1536,3584):
            for chunk in (128,256):
                candidates=[0,n]+[min(n,max(chunk,int(n*f)//chunk*chunk)) for f in (.25,.5,.75)]
                for k in candidates:
                    self.assertTrue(0<=k<=n)
                    self.assertTrue(k in (0,n) or k%chunk==0)

if __name__=='__main__':unittest.main()
