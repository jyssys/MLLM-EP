"""Optional CUDA enclosures and native readiness samples. No scheduling changes."""
import json,os,time,types
from .common import *


def install(cfg):
    if not cfg.get('discovery_event_observer'):return
    import torch
    from vllm_rlt.engine.stage_engine import StageEngine
    from vllm_rlt.worker.model_runner import ModelRunner
    from vllm_rlt.request import RequestPhase as P
    initialize,execute,step,close=StageEngine.__init__,ModelRunner._execute_batch,StageEngine._step,StageEngine.close
    old_ready=StageEngine._execution_ready
    folder=ROOT/'raw'/cfg['name'];marker=folder/'measurement_begin.json';state={};samples=[];ready=[];host_samples=[];natural=[]
    latest={}
    cuts=set(cfg.get('observed_source_cuts',[1,2,3] if cfg['model']=='ouro' else [24,28,30,31]))
    def initialized(engine,*args,**kwargs):
        result=initialize(engine,*args,**kwargs)
        anchor=torch.cuda.Event(enable_timing=True)
        engine.model_runner.synchronize();a=time.monotonic();anchor.record();anchor.synchronize();b=time.monotonic()
        state.update(engine=engine,role=engine.stage_config.role,anchor=anchor,anchor_host=(a+b)/2,
            anchor_host_bracket=[a,b],next_sample=0.)
        graphs=getattr(engine,'_autonomous_prefill_graphs',None)
        if state['role']=='prefill' and graphs is not None:
            original=graphs.execute
            def observed(owner,call,hidden):
                result=original(call,hidden)
                if not marker.exists() or call.metadata.cu_seqlens_q is None:return result
                metadata=call.metadata
                if not metadata.rows:return result
                depth=metadata.rows[0][1]+1
                if depth not in cuts:return result
                assert all(row[1]+1==depth for row in metadata.rows)
                requests=[]
                for request_id,allocation in metadata.allocations:
                    positions=[row[2] for row in metadata.rows if row[0] is allocation]
                    transfer=engine.connector.transfers.get(request_id)
                    if transfer is None:continue
                    request=transfer.request
                    if not 0<=request.sampling_params.seed-cfg['seed']<cfg['count']:continue
                    if positions!=list(range(len(request.prompt_token_ids))):continue
                    requests.append(dict(request_id=request_id,seed=request.sampling_params.seed,tokens=len(positions)))
                if requests:
                    end=torch.cuda.Event(enable_timing=True);end.record(torch.cuda.current_stream(owner.cache.device))
                    samples.append((None,end,dict(kind='SOURCE_LOOP_CUT_END',depth=depth,requests=requests,
                        host_submission=time.monotonic(),stream=int(torch.cuda.current_stream().cuda_stream),
                        state_materialized_for_export=False,graph_hit=call.entry is not None)))
                return result
            graphs.execute=types.MethodType(observed,graphs)
        return result
    def executed(runner,batch,prepared):
        if state.get('role')!='decode' or not marker.exists():return execute(runner,batch,prepared)
        if not cfg.get('discovery_D_execution_events',True):
            begin=time.monotonic()
            try:return execute(runner,batch,prepared)
            finally:host_samples.append(dict(kind='D_HOST_MODEL_SUBMISSION',host_start=begin,host_end=time.monotonic(),
                operation=batch.operation.value,loops=batch.loop_count,rows=batch.num_rows,
                requests=[dict(request_id=s.request.request_id,seed=s.request.sampling_params.seed,
                    token_start=s.token_start,token_count=s.token_count,depth=s.depth,
                    generated_tokens=len(s.request.generated_token_ids)) for s in batch.spans],
                scope='Host submission only. Not GPU kernel time or a CUDA Event enclosure.'))
        stream=torch.cuda.current_stream(runner.device);a,b=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
        meta=dict(kind='D_MODEL_EXECUTION_ENCLOSURE',operation=batch.operation.value,loops=batch.loop_count,
            host_submission=time.monotonic(),stream=int(stream.cuda_stream),rows=batch.num_rows,
            requests=[dict(request_id=s.request.request_id,seed=s.request.sampling_params.seed,
                token_start=s.token_start,token_count=s.token_count,depth=s.depth,
                generated_tokens=len(s.request.generated_token_ids)) for s in batch.spans])
        a.record(stream)
        try:return execute(runner,batch,prepared)
        finally:b.record(stream);samples.append((a,b,meta))
    def checked(engine,request):
        result=old_ready(engine,request)
        if marker.exists() and state.get('engine') is engine:
            now=time.monotonic();previous=latest.get(request.request_id)
            if previous is None or now-previous['time']>=.05 or previous['ready']!=bool(result) or previous['phase']!=request.phase.value:
                row=dict(time=now,role=state['role'],request_id=request.request_id,seed=request.sampling_params.seed,
                    phase=request.phase.value,ready=bool(result),generated_tokens=len(request.generated_token_ids))
                natural.append(row);latest[request.request_id]=row
        return result
    def stepped(engine,*args,**kwargs):
        now=time.monotonic()
        if marker.exists() and now>=state['next_sample']:
            state['next_sample']=now+.05;pending={i.request.request_id for t in engine._inflight for i in t.batch.requests}
            scheduler=engine.scheduler;queues=[]
            for phase in (P.PRELUDE,P.RECURRENT,P.CODA,P.PREFILL,P.PREFILLING_PRELUDE,P.PREFILLING_RECURRENT):
                for rid in list(scheduler.queues[phase]):
                    request=scheduler.requests[rid]
                    last=latest.get(rid)
                    is_ready=(bool(scheduler.execution_ready is None or scheduler.execution_ready(request))
                        if not cfg.get('discovery_natural_readiness') else None)
                    queues.append(dict(request_id=rid,seed=request.sampling_params.seed,phase=phase.value,
                        generated_tokens=len(request.generated_token_ids),inflight=rid in pending,
                        native_execution_ready=is_ready,last_natural_ready_decision=last,
                        readiness_query_scope='Natural scheduler callbacks only' if cfg.get('discovery_natural_readiness') else 'Additional diagnostic predicate query'))
            cache=engine.cache_manager
            ready.append(dict(time=now,role=state['role'],queues=queues,KV_total_blocks=cache.num_blocks,
                KV_free_blocks=cache.num_free_blocks,inflight_requests=sorted(pending)))
        return step(engine,*args,**kwargs)
    def closed(engine):
        engine.model_runner.synchronize();rows=[]
        for begin,end,meta in samples:
            rows.append(dict(meta,gpu_end_host_estimate=state['anchor_host']+state['anchor'].elapsed_time(end)/1000,
                gpu_start_host_estimate=state['anchor_host']+state['anchor'].elapsed_time(begin)/1000 if begin is not None else None,
                CUDA_event_enclosure_ms=begin.elapsed_time(end) if begin is not None else None))
        write(folder/f'discovery_events_{state["role"]}.json',dict(role=state['role'],anchor_host_bracket=state['anchor_host_bracket'],
            rows=rows,readiness_samples=ready,natural_readiness=natural,host_submissions=host_samples,
            D_execution_events_enabled=cfg.get('discovery_D_execution_events',True),
            scope='CUDA Event enclosures and Native readiness observations. They are not exclusive kernels or physical idle. Source events show completed recurrent depth, not a retained export snapshot. Host mapping retains the startup bracket uncertainty.'))
        return close(engine)
    StageEngine.__init__,ModelRunner._execute_batch,StageEngine._step,StageEngine.close=initialized,executed,stepped,closed
    if cfg.get('discovery_natural_readiness'):StageEngine._execution_ready=checked
