"""Observe D operation enclosures. No claim of exclusive CUDA kernel time."""
import os,time
from .common import ROOT,write


def install(cfg):
    import torch
    from vllm_rlt.engine.stage_engine import StageEngine
    from vllm_rlt.worker.model_runner import ModelRunner
    from ..dynamic_loop_split_serving.service_timing import calibrated_device_time
    initialize,close=StageEngine.__init__,StageEngine.close
    execute,submit=ModelRunner._execute,ModelRunner.submit
    ready,schedule=StageEngine._execution_ready,StageEngine._schedule
    marker=ROOT/'raw'/cfg['name']/'measurement_begin.json'

    def active(state):
        if not state['enabled']:state['enabled']=marker.exists()
        return state['enabled']

    def initialized(engine,*args,**kwargs):
        result=initialize(engine,*args,**kwargs)
        if engine.stage_config.role!='decode':return result
        runner=engine.model_runner
        stream=runner.core_stream or torch.cuda.current_stream(runner.device)
        choices=[]
        for _ in range(5):
            e=torch.cuda.Event(enable_timing=True);a=time.monotonic();e.record(stream);e.synchronize();b=time.monotonic()
            choices.append((b-a,e,dict(lower=a,upper=b,midpoint=(a+b)/2)))
        _,anchor,bounds=min(choices,key=lambda x:x[0])
        state=dict(anchor=anchor,bounds=bounds,events=[],schedules=[],ready_changes=[],
            ready_last={},host_submit=None,last_device_end=None,enabled=False,engine=engine)
        if cfg.get('diagnosis_kernel_tail'):
            from .kernel_tail import KernelTail
            state['kernel_tail']=KernelTail(cfg)
        runner._tail_diagnostic=state;engine._tail_diagnostic=state
        return result

    def is_ready(engine,request):
        value=ready(engine,request)
        state=getattr(engine,'_tail_diagnostic',None)
        if state is not None and active(state):
            key=request.request_id
            signature=(request.phase.value,value,len(getattr(engine,'net_pending_coda',[])))
            now=time.monotonic()
            previous=state['ready_last'].get(key)
            if previous is None or previous['signature']!=signature:
                state['ready_changes'].append(dict(time=now,request_id=key,seed=request.sampling_params.seed,
                    phase=request.phase.value,ready=value,pending_coda=signature[2]))
            state['ready_last'][key]=dict(signature=signature,time=now)
        return value

    def scheduled(engine,*args,**kwargs):
        begin=time.monotonic();result=schedule(engine,*args,**kwargs)
        state=getattr(engine,'_tail_diagnostic',None)
        if state is not None and result is not None and active(state):
            state['schedules'].append(dict(start=begin,end=time.monotonic(),phase=result.phase.value,
                queued_after_selection={phase.value:len(queue) for phase,queue in engine.scheduler.queues.items()},
                pending_coda=len(getattr(engine,'net_pending_coda',[])),
                chosen=[dict(request_id=i.request.request_id,seed=i.request.sampling_params.seed,
                    last_native_ready_test=state['ready_last'].get(i.request.request_id)) for i in result.requests]))
        return result

    def submitted(runner,*args,**kwargs):
        state=getattr(runner,'_tail_diagnostic',None)
        if state is None:return submit(runner,*args,**kwargs)
        begin=time.monotonic();cpu=time.thread_time();first=len(state['events'])
        try:return submit(runner,*args,**kwargs)
        finally:
            end=time.monotonic();elapsed=1000*(time.thread_time()-cpu)
            for _,_,row in state['events'][first:]:
                row.update(host_submit_begin=begin,host_submit_end=end,host_submit_thread_CPU_ms=elapsed)

    def observed(runner,scheduled_batch,prepared):
        state=getattr(runner,'_tail_diagnostic',None)
        if state is None or not active(state):return execute(runner,scheduled_batch,prepared)
        batch=prepared.batch;stream=torch.cuda.current_stream(runner.device)
        if 'kernel_tail' in state:state['kernel_tail'].observe(state['engine'],batch)
        before=time.monotonic()
        if cfg.get('diagnosis_serialize') and state['last_device_end'] is not None:
            stream.wait_event(state['last_device_end'])
        a,b=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
        a.record(stream)
        result=execute(runner,scheduled_batch,prepared)
        b.record(stream);state['last_device_end']=b
        rows=[dict(request_id=s.request.request_id,seed=s.request.sampling_params.seed,
            phase=s.request.phase.value,start=s.token_start,tokens=s.token_count,depth=s.depth,
            generated=len(s.request.generated_token_ids)) for s in batch.spans]
        state['events'].append((a,b,dict(host_execute_begin=before,host_execute_end=time.monotonic(),
            stream=int(stream.cuda_stream),operation=batch.operation.value,rows=batch.num_rows,
            loops=batch.loop_count,prompt=any(s.chunk_start is not None for s in batch.spans),
            requests=rows,safe_budget_ms=None,admitted_prediction_ms=None,
            admission_scope='Static native continuation. No online safe-budget admission runs.',
            serialized=bool(cfg.get('diagnosis_serialize')))))
        return result

    def closed(engine):
        state=getattr(engine,'_tail_diagnostic',None)
        if state is not None:
            runner=engine.model_runner;runner.synchronize()
            endpoint=torch.cuda.Event(enable_timing=True)
            stream=runner.core_stream or torch.cuda.current_stream(runner.device)
            low=time.monotonic();endpoint.record(stream);endpoint.synchronize();high=time.monotonic()
            bounds=dict(lower=low,upper=high,midpoint=(low+high)/2)
            total=state['anchor'].elapsed_time(endpoint);rows=[]
            for a,b,metadata in state['events']:
                left,right=state['anchor'].elapsed_time(a),state['anchor'].elapsed_time(b)
                rows.append(dict(metadata,begin=calibrated_device_time(state['bounds'],bounds,total,left),
                    end=calibrated_device_time(state['bounds'],bounds,total,right),
                    device_begin_offset_ms=left,device_end_offset_ms=right,
                    device_enclosure_ms=a.elapsed_time(b)))
            if 'kernel_tail' in state:state['kernel_tail'].close()
            write(ROOT/'raw'/cfg['name']/f'decode_diagnosis_{os.getpid()}.json',dict(
                role='decode',rows=rows,schedules=state['schedules'],ready_changes=state['ready_changes'],
                clock=dict(start=state['bounds'],end=bounds,elapsed_device_ms=total,
                    endpoint_drift_ms=1000*(bounds['midpoint']-state['bounds']['midpoint'])-total),
                scope='CUDA Event operation enclosures, not kernel unions or exclusive kernel times. Ready transitions record Native tests only. Queue lengths include blocked requests. No new readiness tests run. Serialization is a counterfactual intervention, not an optimized serving method.'))
        return close(engine)
    StageEngine.__init__,StageEngine.close=initialized,closed
    StageEngine._execution_ready,StageEngine._schedule=is_ready,scheduled
    ModelRunner._execute,ModelRunner.submit=observed,submitted
