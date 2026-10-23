"""Pretrained lifecycle and genuinely executed heterogeneous-frontier probes."""
import gc
from dataclasses import replace
from .common import write
from .experiments import new_engine,inputs,generate

def run_job(model,job,target):
    from vllm_rlt import SamplingParams
    from vllm_rlt.request import RequestPhase
    results=[]
    for desired in (1,2,4):
        e=new_engine(model,4,desired,job)
        params=SamplingParams(max_tokens=3,max_loops=4,exit_threshold=1.,temperature=0.,ignore_eos=True)
        ids=inputs(256,3)
        e.add_request('reused',ids[0],params)
        # Abort after a real submission, not an unadmitted placeholder.
        for _ in range(400):
            e.step()
            r=e.scheduler.requests.get('reused')
            if r and r.phase in (RequestPhase.RECURRENT,RequestPhase.CODA):break
        e.abort_request('reused')
        assert 'reused' not in e.scheduler.requests
        e.add_request('reused',ids[1],params)
        actual=None
        while e.has_unfinished_requests():
            for output in e.step():actual=output
        assert actual and actual.exit_depths==[4,4,4]
        assert e.cache_manager.num_used_blocks==0
        # Build real decode depth0 frontiers with native prelude and full prompt.
        native_ready=e.scheduler.execution_ready
        released=[False]
        def initial_ready(request):
            if len(e.scheduler.requests)==len(ids) and all(r.generated_token_ids for r in e.scheduler.requests.values()):
                released[0]=True
            return released[0] or request.num_scheduled_outputs==0
        e.scheduler.execution_ready=initial_ready
        for i,tokens in enumerate(ids):e.add_request(str(i),tokens,params)
        while any(not r.generated_token_ids for r in e.scheduler.requests.values()):e.step()
        e.scheduler.execution_ready=native_ready
        while any(r.phase==RequestPhase.PRELUDE for r in e.scheduler.requests.values()):e.step()
        # Advance members to [0,1,3] by ACTUAL single-loop submissions.
        for rid,count in (('1',1),('2',3)):
            for _ in range(count):
                batch=e.scheduler.take_phases((RequestPhase.RECURRENT,),eligible=lambda r:r.request_id==rid)
                batch=replace(batch,requests=[replace(r,loop_count=1) for r in batch.requests])
                ticket=e.executor.submit(batch)
                if job.get('async',True):
                    update=e.scheduler.update_after_submission(batch,completion=ticket.event,request_events=e.model_runner.events)
                    e._apply_update(update)
                else:e._update(batch,ticket.collect() if ticket.has_result else None)
        e.model_runner.synchronize()
        states={r.request_id:r.loops_done for r in e.scheduler.requests.values()}
        assert states=={'0':0,'1':1,'2':3},states
        batch=e.scheduler.schedule()
        assert sorted(r.depth for r in batch.requests)==[0,1,3]
        assert all(r.loop_count==1 for r in batch.requests)
        ticket=e.executor.submit(batch)
        if job.get('async',True):
            e._apply_update(e.scheduler.update_after_submission(batch,completion=ticket.event,request_events=e.model_runner.events))
        else:e._update(batch,ticket.collect() if ticket.has_result else None)
        outputs={}
        while e.has_unfinished_requests():
            for output in e.step():outputs[output.request_id]=output
        assert len(outputs)==3 and all(o.exit_depths==[4,4,4] for o in outputs.values())
        assert e.cache_manager.num_used_blocks==0
        item=dict(desired_D=desired,actual_frontiers=states,selected_common_k=1,
            cancel_reuse_tokens=list(actual.token_ids),heterogeneous_outputs={r:list(o.token_ids) for r,o in outputs.items()},
            zero_used_blocks_after_drain=True,evidence='PRETRAINED_LIVE_LIFECYCLE_NOT_TASK_QUALITY')
        write(target/f'lifecycle_D{desired}.json',item);results.append(item)
        del native_ready,batch,ticket,actual,initial_ready
        e.close();del e;gc.collect();__import__('torch').cuda.empty_cache()
    return dict(results=results)
