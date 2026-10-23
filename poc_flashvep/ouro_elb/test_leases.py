"""CPU software invariants. These fixtures are not pretrained quality evidence."""
from dataclasses import replace
from types import SimpleNamespace
import unittest
from vllm_rlt import SchedulerConfig, SamplingParams
from vllm_rlt.core.scheduler import Scheduler, SchedulerOutput, ScheduledRequest
from vllm_rlt.request import Request, RequestPhase
from vllm_rlt.worker.executor import build_execution_batch
from vllm_rlt.worker.output import ModelRunnerOutput

class LeaseTests(unittest.TestCase):
    def fixture(self, depths, desired=4, asynchronous=False):
        cache=SimpleNamespace(max_loops=4)
        scheduler=Scheduler(SchedulerConfig(decode_loop_group_size=desired),cache,
            total_ut_steps=4,exit_mode='ouro_delayed' if asynchronous else 'ouro')
        rows=[]
        for i,depth in enumerate(depths):
            r=Request(str(i),[1,2],SamplingParams(max_tokens=3,max_loops=4,exit_threshold=1.),
                phase=RequestPhase.RECURRENT,loops_done=depth,generated_token_ids=[3])
            scheduler.requests[r.request_id]=r
            rows.append(ScheduledRequest(r,r.position,depth=depth,phase=RequestPhase.RECURRENT))
        scheduler.policy=SimpleNamespace(schedule=lambda *a,**kw:SchedulerOutput(RequestPhase.RECURRENT,rows))
        return scheduler,rows

    def test_legal_common_ceiling(self):
        for depths in ([0,0],[0,1],[0,2,3],[1,3],[2,2]):
            for desired in (1,2,3,4):
                scheduler,rows=self.fixture(depths,desired)
                batch=scheduler.schedule()
                k=min(desired,*(4-d for d in depths))
                self.assertEqual([r.loop_count for r in batch.requests],[k]*len(rows))
                build_execution_batch(batch,4).validate_depths(4)

    def test_no_prelude_or_coda_group(self):
        scheduler,rows=self.fixture([0],4)
        for phase in (RequestPhase.PRELUDE,RequestPhase.CODA):
            scheduler.policy.schedule=lambda *a,**kw:SchedulerOutput(phase,[replace(rows[0],phase=phase)])
            self.assertEqual(build_execution_batch(scheduler.schedule(),4).loop_count,0)

    def test_mixed_remains_one(self):
        scheduler,rows=self.fixture([0,1],4)
        mixed=[rows[0],replace(rows[1],phase=RequestPhase.PREFILLING_RECURRENT,chunk_start=0)]
        scheduler.policy.schedule=lambda *a,**kw:SchedulerOutput(None,mixed)
        self.assertEqual(build_execution_batch(scheduler.schedule(),4).loop_count,1)

    def test_reject_invalid_direct_lease(self):
        scheduler,rows=self.fixture([0,3],2)
        batch=SchedulerOutput(RequestPhase.RECURRENT,[replace(r,loop_count=2) for r in rows])
        with self.assertRaises(ValueError):build_execution_batch(batch,4)

    def test_reject_adaptive(self):
        scheduler,rows=self.fixture([0],2)
        rows[0].request.sampling_params=replace(rows[0].request.sampling_params,exit_threshold=.5)
        with self.assertRaises(ValueError):scheduler.schedule()

    def test_advances_each_real_loop(self):
        for asynchronous in (False,True):
            for depths,desired in (([0,0],2),([0],4),([1],3),([2,3],4)):
                scheduler,rows=self.fixture(depths,desired,asynchronous)
                batch=scheduler.schedule();k=batch.requests[0].loop_count
                if asynchronous:scheduler.update_after_submission(batch)
                else:scheduler.update_from_output(batch,ModelRunnerOutput(gate_scores=[0.]*len(rows)))
                for r,d in zip(rows,depths):
                    self.assertEqual(r.request.loops_done,d+k)
                    self.assertEqual(r.request.phase,RequestPhase.CODA if d+k==4 else RequestPhase.RECURRENT)

    def test_stale_request_identity(self):
        scheduler,rows=self.fixture([0],4,True);batch=scheduler.schedule()
        replacement=Request('0',[5],rows[0].request.sampling_params)
        scheduler.requests['0']=replacement
        scheduler.update_after_submission(batch)
        self.assertEqual(replacement.loops_done,0)

    def test_noninteger_group(self):
        for invalid in (0,5,-1):
            with self.assertRaises(ValueError):SchedulerConfig(decode_loop_group_size=invalid)

if __name__=='__main__':unittest.main()
