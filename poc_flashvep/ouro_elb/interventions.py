"""Legal recurrent-ready interventions. Not API-arrival serving claims."""
import gc
import hashlib
import time
from dataclasses import replace
import torch
from .common import write,guard
from .experiments import new_engine,inputs,generate

def state_hash(e):
    e.model_runner.synchronize()
    values=[]
    for rid,r in sorted(e.scheduler.requests.items()):
        values.append(rid.encode());values.append(r.hidden_state.cpu().view(torch.uint8).numpy().tobytes())
        for depth in range(4):
            for layer in range(24):
                k,v=e.cache_manager.read(layer,rid,depth,len(r.prompt_token_ids))
                for tensor in (k[[0,-1]],v[[0,-1]]):values.append(tensor.cpu().view(torch.uint8).numpy().tobytes())
    return hashlib.sha256(b''.join(values)).hexdigest()

def run_job(model,job,target):
    from vllm_rlt import SamplingParams
    from vllm_rlt.request import RequestPhase
    results=[]
    for cell in job['cells']:
        P,old,new=cell['P'],cell['old'],cell['new'];N=old+new
        ids=inputs(P,N);reference=None
        for action in ('KEEP4','YIELD1','NO_REFORM1'):
            # Sync is a declared diagnostic lane, not a native-async headline.
            e=new_engine(model,N,1,dict(job,**{'async':False}))
            # Both strategies can create any row width up to old+new.
            # Warm each graph geometry before either measured action.
            for width in range(1,N+1):generate(e,ids[:width],cap=4)
            params=SamplingParams(max_tokens=2,min_loops=4,max_loops=4,exit_threshold=1.,temperature=0.,ignore_eos=True)
            for i,tokens in enumerate(ids):e.add_request(str(i),tokens,params)
            native_ready=e.scheduler.execution_ready
            e.scheduler.execution_ready=lambda r: not r.generated_token_ids
            while any(not r.generated_token_ids for r in e.scheduler.requests.values()):e.step()
            e.scheduler.execution_ready=native_ready
            while any(r.phase==RequestPhase.PRELUDE for r in e.scheduler.requests.values()):e.step()
            assert all(r.phase==RequestPhase.RECURRENT and r.loops_done==0 for r in e.scheduler.requests.values())
            fingerprint=state_hash(e)
            if reference is None:reference=fingerprint
            assert reference==fingerprint,'Intervention source state is not identical'
            # Make the old cohort's first lease exclude the initially held new rows.
            old_ids=set(map(str,range(old)))
            initial=e.scheduler.take_phases((RequestPhase.RECURRENT,),eligible=lambda r:r.request_id in old_ids)
            k=4 if action=='KEEP4' else 1
            initial=replace(initial,requests=[replace(r,loop_count=k) for r in initial.requests])
            runner=e.model_runner;execute=runner.graphs.execute
            anchor=torch.cuda.Event(enable_timing=True);eligible_event=torch.cuda.Event(enable_timing=True)
            calls=[0];timeline=[]
            def mark(call,hidden):
                result=execute(call,hidden);calls[0]+=1
                if calls[0]==1:eligible_event.record(torch.cuda.current_stream())
                return result
            runner.graphs.execute=mark
            captures_before=runner.graphs.captures
            anchor.record();anchor.synchronize();start=time.perf_counter()
            ticket=e.executor.submit(initial);e._update(initial,ticket.collect() if ticket.has_result else None)
            runner.graphs.execute=execute
            if action=='NO_REFORM1':
                # Generic fixed-membership D1 control. It performs the same old-row
                # priority as KEEP4 without a multi-loop submission.
                for _ in range(3):
                    batch=e.scheduler.take_phases((RequestPhase.RECURRENT,),eligible=lambda r:r.request_id in old_ids)
                    assert batch and all(row.depth<4 for row in batch.requests)
                    ticket=e.executor.submit(batch)
                    e._update(batch,ticket.collect() if ticket.has_result else None)
            original_schedule=e.scheduler.schedule
            def select(**kwargs):
                batch=original_schedule(**kwargs)
                if batch:
                    timeline.append(dict(t=time.perf_counter()-start,phase=str(batch.phase),rows=[dict(id=r.request.request_id,depth=r.depth) for r in batch.requests]))
                return batch
            e.scheduler.schedule=select
            outputs={};completion={}
            while e.has_unfinished_requests():
                for output in e.step():
                    if len(output.token_ids)==2:
                        outputs[output.request_id]=list(output.token_ids)
                        completion[output.request_id]=time.perf_counter()-start
            runner.synchronize();eligible_ms=anchor.elapsed_time(eligible_event)
            item=dict(P=P,old=old,new=new,action=action,gpu=guard(),state_sha256=fingerprint,
                eligibility='Precomputed new-row PRELUDE state becomes eligible at old cohort R1 GPU event. Not API arrival.',
                eligible_ms=eligible_ms,completion_ms={r:1000*v for r,v in completion.items()},
                old_completion_ms=max(1000*completion[r] for r in old_ids),
                new_completion_ms=max(1000*v for r,v in completion.items() if r not in old_ids),
                all_completion_ms=1000*max(completion.values()),outputs=outputs,timeline=timeline,
                graph_captures_during_intervention=runner.graphs.captures-captures_before,
                lane='LIVE_MEASURED_CONTROLLED_SYNC_COMPONENT',new_requests_finish_at_depth=4)
            write(target/f'intervention_P{P}_O{old}_N{new}_{action}.json',item);results.append(item)
            assert item['graph_captures_during_intervention']==0,'Unwarmed graph shape confounds the intervention'
            e.scheduler.schedule=original_schedule
            del select,original_schedule,execute,mark,runner,ticket,initial,native_ready
            e.close();del e;gc.collect();torch.cuda.empty_cache()
    return dict(results=results)
