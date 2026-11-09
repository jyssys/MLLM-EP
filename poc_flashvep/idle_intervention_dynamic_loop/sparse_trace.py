"""Optional event enclosures. Unsampled intervals are not physical idle time."""
import os
import time
from .common import ROOT,write


def install(cfg):
    import torch
    from vllm_rlt.engine.stage_engine import StageEngine
    from vllm_rlt.worker.model_runner import ModelRunner
    from ..dynamic_loop_split_serving.service_timing import calibrated_device_time,prompt_span
    initialize,close,execute=StageEngine.__init__,StageEngine.close,ModelRunner._execute
    folder=ROOT/'raw'/cfg['name'];marker=folder/'measurement_begin.json'

    def initialized(engine,*args,**kwargs):
        result=initialize(engine,*args,**kwargs);runner=engine.model_runner
        stream=runner.core_stream or torch.cuda.current_stream(runner.device)
        brackets=[]
        for _ in range(5):
            event=torch.cuda.Event(enable_timing=True);low=time.monotonic()
            event.record(stream);event.synchronize();high=time.monotonic()
            brackets.append((high-low,event,dict(lower=low,upper=high,midpoint=(low+high)/2)))
        _,anchor,bounds=min(brackets,key=lambda row:row[0])
        runner._idle_sparse=dict(anchor_event=anchor,anchor=bounds,samples=[],counter=0,
            measured=False,record_host_s=0.,role=engine.stage_config.role)
        return result

    def observed(runner,scheduled_batch,prepared):
        state=runner._idle_sparse
        if not state['measured']:state['measured']=marker.exists()
        if not state['measured']:return execute(runner,scheduled_batch,prepared)
        batch=prepared.batch;prompt=any(prompt_span(batch,span) for span in batch.spans)
        state['counter']+=1
        # A prime period avoids locking the sample to Huginn's 32-depth cycle.
        if not prompt and state['counter']%cfg.get('sparse_decode_period',31):
            return execute(runner,scheduled_batch,prepared)
        stream=torch.cuda.current_stream(runner.device)
        begin,end=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
        start=time.perf_counter();host=time.monotonic();begin.record(stream)
        state['record_host_s']+=time.perf_counter()-start
        result=execute(runner,scheduled_batch,prepared)
        start=time.perf_counter();end.record(stream);state['record_host_s']+=time.perf_counter()-start
        state['samples'].append((begin,end,dict(prompt=prompt,operation=batch.operation.value,
            rows=batch.num_rows,loops=batch.loop_count,host_submit=host,
            requests=[dict(tid=s.request.request_id,seed=s.request.sampling_params.seed,
                start=s.token_start,count=s.token_count,depth=s.depth) for s in batch.spans])))
        return result

    def closed(engine):
        runner=engine.model_runner;runner.synchronize();state=runner._idle_sparse
        stream=runner.core_stream or torch.cuda.current_stream(runner.device)
        endpoint=torch.cuda.Event(enable_timing=True);low=time.monotonic()
        endpoint.record(stream);endpoint.synchronize();high=time.monotonic()
        end_anchor=dict(lower=low,upper=high,midpoint=(low+high)/2)
        anchor=state['anchor_event'];total=anchor.elapsed_time(endpoint);rows=[]
        for begin,end,metadata in state['samples']:
            a,b=anchor.elapsed_time(begin),anchor.elapsed_time(end)
            rows.append(dict(metadata,begin=calibrated_device_time(state['anchor'],end_anchor,total,a),
                end=calibrated_device_time(state['anchor'],end_anchor,total,b),
                CUDA_enclosure_ms=begin.elapsed_time(end)))
        write(folder/f'sparse_device_{os.getpid()}.json',dict(role=state['role'],rows=rows,
            anchor=state['anchor'],end_anchor=end_anchor,event_record_host_s=state['record_host_s'],
            endpoint_drift_ms=1000*(end_anchor['midpoint']-state['anchor']['midpoint'])-total,
            Decode_sample_period=cfg.get('sparse_decode_period',31),
            evidence='MEASURED_CUDA_EVENT_ENCLOSURES_NOT_KERNEL_UNION',
            scope='All measured prompt submissions and sampled Decode submissions. Unrecorded intervals are unknown. No timed-submission synchronization. Endpoint brackets do not bound nonlinear clock drift.'))
        return close(engine)
    StageEngine.__init__,StageEngine.close,ModelRunner._execute=initialized,closed,observed
