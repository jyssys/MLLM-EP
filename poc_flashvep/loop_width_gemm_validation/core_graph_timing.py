"""Observe the actual packed graph instance, including BulkWrittenGraphs."""
import os,types
from .common import *


def install(cfg):
    if not cfg.get('observe_core_graph_instance'):return
    assert cfg['calibration_serial'] and cfg['observe_device_execution']
    import torch
    from vllm_rlt.worker.model_runner import ModelRunner
    from vllm_rlt.engine.stage_engine import StageEngine
    initialize,execute,close=StageEngine.__init__,ModelRunner._execute_batch,StageEngine.close
    current=[None];samples=[];marker=ROOT/'raw'/cfg['name']/'measurement_begin.json'
    def initialized(engine,*args,**kwargs):
        result=initialize(engine,*args,**kwargs)
        graphs=getattr(engine,'_autonomous_prefill_graphs',None)
        if graphs is not None:
            native=graphs.execute
            def observed(owner,call,hidden):
                if current[0] is None:return native(call,hidden)
                context=current[0];stream=torch.cuda.current_stream()
                begin,end=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
                begin.record(stream);output=native(call,hidden);end.record(stream)
                samples.append((begin,end,dict(context,component='packed_recurrent_graph',
                    actual_depth=context['batch_depth']+context['call_index'],
                    graph_fallback=call.entry is None,graph_class=type(owner).__name__)))
                context['call_index']+=1
                return output
            graphs.execute=types.MethodType(observed,graphs)
        return result
    def executed(runner,batch,prepared):
        previous=current[0]
        if marker.exists() and len(batch.spans)==1:
            span=batch.spans[0]
            if not span.request.generated_token_ids and span.token_count>1:
                current[0]=dict(seed=span.request.sampling_params.seed,tokens=span.token_count,
                    start=span.token_start,batch_depth=span.depth,planned_loops=batch.loop_count,
                    operation=batch.operation.value,call_index=0)
        try:return execute(runner,batch,prepared)
        finally:current[0]=previous
    def closed(engine):
        engine.model_runner.synchronize()
        write(ROOT/'raw'/cfg['name']/f'core_graph_execution_{os.getpid()}.json',dict(
            role=engine.stage_config.role,rows=[dict(meta,ms=a.elapsed_time(b)) for a,b,meta in samples],
            scope='Measured legal packed-core graph enclosures. Includes output clones and host launch gaps. Excludes separate Prelude, prompt Coda, graph-input preparation, and transfer. Not exclusive kernel union or clean serving.'))
        return close(engine)
    StageEngine.__init__=initialized;ModelRunner._execute_batch=executed;StageEngine.close=closed
