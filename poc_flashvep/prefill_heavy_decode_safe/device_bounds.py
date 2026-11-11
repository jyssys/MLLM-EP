"""Conservative GPU-event enclosures. Validate kernel coverage separately."""
import os,time
from .common import *


def install(cfg):
    if not cfg.get('discovery_D_device_bounds'):return
    import torch
    from vllm_rlt.engine.stage_engine import StageEngine
    from vllm_rlt.worker.model_runner import ModelRunner
    from ..native_loop_2d_oracle import native_path
    old_init,old_close=StageEngine.__init__,StageEngine.close
    state={};samples=[];marker=ROOT/'raw'/cfg['name']/'measurement_begin.json'
    def initialized(engine,*args,**kwargs):
        result=old_init(engine,*args,**kwargs)
        if engine.stage_config.role!='decode':return result
        runner=engine.model_runner;torch.cuda.synchronize(runner.device)
        anchor=torch.cuda.Event(enable_timing=True);a=time.monotonic()
        anchor.record(torch.cuda.default_stream(runner.device));anchor.synchronize();b=time.monotonic()
        streams=[torch.cuda.default_stream(runner.device)]
        streams.extend(s for s in (runner.core_stream,runner.boundary_stream,runner.copy_stream,runner.readback_stream) if s is not None)
        streams=list({int(s.cuda_stream):s for s in streams}.values())
        state.update(engine=engine,runner=runner,anchor=anchor,anchor_host=(a+b)/2,
            anchor_bracket=[a,b],streams=streams,depth=0)
        return result
    def observed(label,owner,function,*args,**kwargs):
        if not state or not marker.exists():return function(owner,*args,**kwargs)
        engine=owner.engine if hasattr(owner,'engine') else None
        if owner is not state['runner'] and engine is not state['engine']:return function(owner,*args,**kwargs)
        if state['depth']:return function(owner,*args,**kwargs)
        state['depth']+=1;pairs=[];start=time.monotonic()
        try:
            for stream in state['streams']:
                a,b=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
                a.record(stream);pairs.append((stream,a,b))
            return function(owner,*args,**kwargs)
        finally:
            for stream,a,b in pairs:b.record(stream)
            samples.append((label,start,time.monotonic(),pairs));state['depth']-=1
    submit=ModelRunner.submit
    def submitted(runner,*args,**kwargs):return observed('MODEL_RUNNER_SUBMIT',runner,submit,*args,**kwargs)
    ModelRunner.submit=submitted
    Connector=native_path.ACTIVE_CONNECTOR
    for method in ('command','accept_cut'):
        previous=getattr(Connector,method)
        def wrapped(owner,*args,_function=previous,_label=method,**kwargs):
            return observed('NATIVE_CONNECTOR_'+_label.upper(),owner,_function,*args,**kwargs)
        setattr(Connector,method,wrapped)
    def closed(engine):
        if engine.stage_config.role=='decode' and state:
            torch.cuda.synchronize(state['runner'].device)
            end=torch.cuda.Event(enable_timing=True);a=time.monotonic()
            end.record(torch.cuda.default_stream(state['runner'].device));end.synchronize();b=time.monotonic()
            total=state['anchor'].elapsed_time(end);end_host=(a+b)/2;rows=[]
            def host(event):return state['anchor_host']+state['anchor'].elapsed_time(event)/total*(end_host-state['anchor_host'])
            for label,start,finish,pairs in samples:
                for stream,begin,stop in pairs:
                    rows.append(dict(operation=label,host_start=start,host_return=finish,
                        cuda_stream=int(stream.cuda_stream),gpu_start=host(begin),gpu_end=host(stop),
                        CUDA_event_enclosure_ms=begin.elapsed_time(stop)))
            write(ROOT/'raw'/cfg['name']/f'D_device_bounds_{os.getpid()}.json',dict(rows=rows,
                streams=[int(s.cuda_stream) for s in state['streams']],anchor_host_bracket=state['anchor_bracket'],
                end_host_bracket=[a,b],CUDA_anchor_interval_ms=total,
                endpoint_clock_drift_ms=1000*(end_host-state['anchor_host'])-total,
                scope='GPU-event enclosures on every known model stream and the default stream. Host method duration is separate. Enclosures include waits and idle. They are not kernel unions. Kernel coverage, clock padding, and matched latency controls require separate validation.'))
        return old_close(engine)
    StageEngine.__init__,StageEngine.close=initialized,closed
