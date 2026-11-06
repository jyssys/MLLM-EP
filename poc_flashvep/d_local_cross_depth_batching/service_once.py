"""Record first model-stream entry once per request, outside model arithmetic."""
import os
import time
from .common import ROOT,write


def install(cfg):
    if not cfg.get('dlocal_service_once'):return
    import torch
    from vllm_rlt.engine.stage_engine import StageEngine
    from vllm_rlt.worker.model_runner import ModelRunner
    from ..dynamic_loop_split_serving.service_timing import calibrated_device_time
    initialize,execute,closed=StageEngine.__init__,ModelRunner._execute,StageEngine.close
    def anchor(stream):
        event=torch.cuda.Event(enable_timing=True)
        low=time.monotonic();event.record(stream);event.synchronize();high=time.monotonic()
        return event,dict(lower=low,upper=high,midpoint=(low+high)/2)
    def initialized(engine,*args,**kwargs):
        result=initialize(engine,*args,**kwargs)
        runner=engine.model_runner
        pool=[]
        if cfg.get('dlocal_service_preallocate'):
            stream=runner.core_stream or torch.cuda.current_stream(runner.device)
            pool=[torch.cuda.Event(enable_timing=True) for _ in range(1024)]
            for item in pool:item.record(stream)
            stream.synchronize()
        event,clock=anchor(runner.core_stream or torch.cuda.current_stream(runner.device))
        runner._local_first_entry=dict(anchor_event=event,anchor=clock,requests={},pool=pool)
        return result
    def executed(runner,scheduled,prepared):
        state=runner._local_first_entry
        pending=[s.request for s in prepared.batch.spans if s.request.request_id not in state['requests']]
        if pending:
            if cfg.get('dlocal_service_preallocate'):
                assert state['pool'],'First-entry marker pool exhausted'
                event=state['pool'].pop()
            else:event=torch.cuda.Event(enable_timing=True)
            event.record(torch.cuda.current_stream(runner.device))
            for request in pending:
                state['requests'][request.request_id]=dict(begin=event,seed=request.sampling_params.seed,
                    prompt_tokens=len(request.prompt_token_ids))
        return execute(runner,scheduled,prepared)
    def close(engine):
        runner=engine.model_runner;runner.synchronize();state=runner._local_first_entry
        end,clock=anchor(runner.core_stream or torch.cuda.current_stream(runner.device))
        total=state['anchor_event'].elapsed_time(end)
        records=[]
        for rid,row in state['requests'].items():
            records.append(dict(request_id=rid,seed=row['seed'],prompt_tokens=row['prompt_tokens'],
                first_model_stream_entry=calibrated_device_time(state['anchor'],clock,total,
                    state['anchor_event'].elapsed_time(row['begin']))))
        write(ROOT/'raw'/cfg['name']/f'service_device_times_{os.getpid()}.json',dict(role=engine.stage_config.role,
            requests=records,anchor=state['anchor'],end_anchor=clock,CUDA_anchor_interval_ms=total,
            preallocated_markers=bool(cfg.get('dlocal_service_preallocate')),unused_preallocated_markers=len(state['pool']),
            scope='One first model-stream marker per request. No per-loop end marker or active CUPTI collection. Host input preparation precedes this marker. Endpoint clock brackets do not bound nonlinear drift.'))
        return closed(engine)
    StageEngine.__init__,ModelRunner._execute,StageEngine.close=initialized,executed,close
