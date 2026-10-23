import unittest
from types import SimpleNamespace
from ..admission import BudgetAdmission
from ..online import OnlineControl
from ..fairness import attach
from vllm_rlt import SamplingParams,SchedulerConfig
from tests.helpers import make_prefill_engine

class ModelFixture:
    # Synthetic costs verify control flow only, not GPU speed.
    def estimate(self,path,c,g,context,depth):
        return ((.1 if path=='staged_prelude' else 3+g),True)

class AdmissionTests(unittest.TestCase):
    def test_unaffordable_prefill_defers_for_decode_and_has_bounded_escape(self):
        clock=[0.]
        req=SimpleNamespace(request_id='p',phase=SimpleNamespace(value='prefill'))
        dec=SimpleNamespace(request_id='d',phase=SimpleNamespace(value='recurrent'))
        scheduler=SimpleNamespace(execution_ready=None,take_phases=lambda *a,**k:None,
            config=SimpleNamespace(max_num_batched_tokens=2048),requests=dict(p=req,d=dec))
        engine=SimpleNamespace(scheduler=scheduler)
        action=SimpleNamespace(snapshot=lambda *args:(SimpleNamespace(token_count=64),dict(start=0,depth=0)))
        selector=SimpleNamespace(choices=lambda s:(),starvation_ms=250.,budget=10.)
        guard=BudgetAdmission(engine,action,selector,lambda:clock[0])
        try:
            self.assertFalse(guard.ready(req))
            self.assertTrue(guard.ready(dec))
            clock[0]=.251
            self.assertTrue(guard.ready(req))
            self.assertIn('STARVATION',guard.trace[-1]['reason'])
        finally:
            guard.close()

    def test_native_loop_frontiers_under_online_budget_control(self):
        results=[]
        for mode in ('C_ONLY','G_ONLY','JOINT'):
            e=make_prefill_engine(scheduler_config=SchedulerConfig(max_num_seqs=3,max_num_batched_tokens=8,
                prefill_chunk_size=8,prefill_loop_group_size=1))
            attach(e,True)
            control=OnlineControl(e,ModelFixture(),mode,5.,fixed_C=4,yield_penalty_ms=1.)
            for rid,ids in [('a',[1,2,3,4,5,6,7,8,9]),('b',[4,3,2,1,5,6]),('c',[7,8,9])]:
                e.add_request(rid,ids,SamplingParams(max_tokens=4,max_loops=4,exit_threshold=1.,ignore_eos=True))
            outputs={};work={k:0 for k in ('a','b','c')};turns=0
            try:
                while e.has_unfinished_requests():
                    for output in e.step():
                        outputs[output.request_id]=list(output.token_ids)
                    if e.last_schedule:
                        spans=[r for r in e.last_schedule.requests if e.last_schedule.phase_for(r).value=='prefilling_recurrent']
                        self.assertLessEqual(len(spans),1)
                        for r in spans:
                            work[r.request.request_id]+=r.token_count*r.loop_count
                    turns+=1;self.assertLess(turns,500)
                self.assertEqual(work,dict(a=36,b=24,c=12))
                self.assertEqual(e.cache_manager.num_used_blocks,0)
                self.assertTrue(control.selector.trace)
                results.append(outputs)
            finally:
                control.close();e.close()
        self.assertEqual(results[0],results[1]);self.assertEqual(results[0],results[2])

if __name__=='__main__':
    unittest.main()
