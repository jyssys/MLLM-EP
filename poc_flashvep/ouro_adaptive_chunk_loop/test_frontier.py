"""Tiny CPU state-machine tests. These are not pretrained model evidence."""
from dataclasses import replace
import unittest
from vllm_rlt import SamplingParams,SchedulerConfig
from tests.helpers import make_prefill_engine

class FrontierTests(unittest.TestCase):
    def run_case(self,groups,chunk=3,prompt=(1,2,3,4,5,6,7)):
        config=SchedulerConfig(max_num_seqs=2,max_num_batched_tokens=4,prefill_chunk_size=chunk,prefill_loop_group_size=groups[0])
        e=make_prefill_engine(scheduler_config=config)
        e.add_request('probe',list(prompt),SamplingParams(max_tokens=3,max_loops=4,exit_threshold=1.,ignore_eos=True))
        spans=[]
        step=0
        try:
            while e.has_unfinished_requests():
                e.scheduler.config=replace(e.scheduler.config,prefill_loop_group_size=groups[step%len(groups)])
                out=e.step()
                for r in e.last_schedule.requests:
                    if e.last_schedule.phase_for(r).value=='prefilling_recurrent':
                        spans.append((r.token_start,r.token_count,r.depth,r.loop_count))
                for output in out:
                    final=output
                step+=1
                self.assertLess(step,200)
            self.assertEqual(sum(count*g for start,count,d,g in spans),4*len(prompt))
            self.assertTrue(all(depth+g<=4 for start,count,depth,g in spans))
            self.assertTrue(all(d==4 for d in final.exit_depths))
            self.assertEqual(e.model_runner.state_slots,{})
            self.assertEqual(e.cache_manager.num_used_blocks,0)
            return list(final.token_ids),spans
        finally:
            e.close()

    def test_g3_remainder_and_cleanup(self):
        tokens,spans=self.run_case([3])
        self.assertIn(3,[g for start,count,d,g in spans])
        self.assertIn(1,[g for start,count,d,g in spans])

    def test_variable_g_at_legal_frontiers(self):
        tokens,spans=self.run_case([1,2,3,4])
        self.assertGreater(len(spans),0)

    def test_batched_G3_heterogeneous_frontiers(self):
        config=SchedulerConfig(max_num_seqs=3,max_num_batched_tokens=4,prefill_chunk_size=3,prefill_loop_group_size=3)
        e=make_prefill_engine(scheduler_config=config)
        for rid,prompt in [('a',[1,2,3,4,5,6,7]),('b',[3,4,5,6,7]),('c',[7,6,5,4])]:
            e.add_request(rid,prompt,SamplingParams(max_tokens=3,max_loops=4,exit_threshold=1.,ignore_eos=True))
        try:
            steps=0
            while e.has_unfinished_requests():
                e.step()
                groups={r.loop_count for r in e.last_schedule.requests if e.last_schedule.phase_for(r).value=='prefilling_recurrent'}
                self.assertLessEqual(len(groups),1)
                steps+=1
                self.assertLess(steps,200)
            self.assertEqual(e.cache_manager.num_used_blocks,0)
        finally:
            e.close()

if __name__=='__main__':
    unittest.main()
