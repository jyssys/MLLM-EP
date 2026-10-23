import unittest
from vllm_rlt import SamplingParams,SchedulerConfig
from tests.helpers import make_prefill_engine
from ..actions import ChunkActions,Choice
from ..fairness import attach

class NativeTests(unittest.TestCase):
    def run_case(self,completion_fairness=False,variable=False):
        e=make_prefill_engine(scheduler_config=SchedulerConfig(max_num_seqs=3,max_num_batched_tokens=4,
            prefill_chunk_size=4,prefill_loop_group_size=1))
        fairness=attach(e,completion_fairness)
        hook=ChunkActions(e,lambda s:Choice(s['C'] if s['kind']=='CONTINUE_CHUNK' else (3 if s['start']==0 else 4),
            2 if s['start']==0 and variable else 4))
        for rid,ids in [('a',[1,2,3,4,5,6,7,8,9]),('b',[4,3,2,1,5,6]),('c',[7,8,9])]:
            e.add_request(rid,ids,SamplingParams(max_tokens=4,max_loops=4,exit_threshold=1.,ignore_eos=True))
        work={r:0 for r in ('a','b','c')};outputs={};turns=0
        try:
            while e.has_unfinished_requests():
                for output in e.step():
                    outputs[output.request_id]=list(output.token_ids)
                    self.assertTrue(all(d==4 for d in output.exit_depths))
                if e.last_schedule:
                    for r in e.last_schedule.requests:
                        if e.last_schedule.phase_for(r).value=='prefilling_recurrent':
                            work[r.request.request_id]+=r.token_count*r.loop_count
                turns+=1;self.assertLess(turns,300)
            self.assertEqual(work,dict(a=36,b=24,c=12))
            self.assertEqual(e.cache_manager.num_used_blocks,0)
            self.assertEqual(e.model_runner.state_slots,{})
            if variable:
                self.assertTrue(any(r['chosen_G']==2 for r in hook.decisions if r['phase']=='prefilling_recurrent'))
            return outputs
        finally:
            hook.close();e.close()

    def test_disabled_and_enabled_fairness_same_isolated_outputs(self):
        self.assertEqual(self.run_case(),self.run_case(True))

    def test_dynamic_C_and_G_preserve_native_frontiers(self):
        self.assertEqual(self.run_case(variable=False),self.run_case(True,True))

if __name__=='__main__':
    unittest.main()
