"""Serial-only event enclosures around legal recurrent graph execution."""
import os,types
from .common import *


def supported_boundaries(model):
    return [(name,label) for name,label in (('prelude_prepared','prelude'),('coda_prepared','prompt_coda'))
        if callable(getattr(model,name,None))]


def install(cfg):
    if not (cfg.get('calibration_serial') and cfg.get('observe_device_execution')):return
    import torch
    from vllm_rlt.worker.model_runner import ModelRunner
    from vllm_rlt.worker.cuda_graph import RecurrentGraphs
    from vllm_rlt.models import AutoModelForCausalLM
    from vllm_rlt.engine.stage_engine import StageEngine
    execute,graph,load,close=ModelRunner._execute_batch,RecurrentGraphs.execute,AutoModelForCausalLM.from_pretrained,StageEngine.close
    current=[None];samples=[];marker=ROOT/'raw'/cfg['name']/'measurement_begin.json'

    def record(kind,function,*args,**kwargs):
        context=current[0]
        if context is None:return function(*args,**kwargs)
        stream=torch.cuda.current_stream()
        begin,end=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
        begin.record(stream);result=function(*args,**kwargs);end.record(stream)
        samples.append((begin,end,dict(context,component=kind)))
        return result

    def executed(runner,batch,prepared):
        previous=current[0]
        if marker.exists() and len(batch.spans)==1:
            span=batch.spans[0]
            if not span.request.generated_token_ids and span.token_count>1:
                current[0]=dict(seed=span.request.sampling_params.seed,tokens=span.token_count,
                    start=span.token_start,batch_depth=span.depth,core_calls_so_far=0,planned_loops=batch.loop_count,
                    operation=batch.operation.value)
        try:return execute(runner,batch,prepared)
        finally:current[0]=previous

    def graphed(owner,call,hidden):
        if current[0] is None:return graph(owner,call,hidden)
        current[0]['graph_fallback']=call.entry is None
        current[0]['actual_depth']=current[0]['batch_depth']+current[0]['core_calls_so_far']
        result=record('recurrent_graph',graph,owner,call,hidden)
        current[0]['core_calls_so_far']+=1
        return result

    def loaded(cls,*args,**kwargs):
        model=load(*args,**kwargs)
        methods=supported_boundaries(model)
        assert cfg['model']!='huginn' or len(methods)==2
        for name,label in methods:
            base=getattr(model,name)
            def observed(self,x,*args,_base=base,_label=label,**kwargs):
                if len(x)<=1:return _base(x,*args,**kwargs)
                return record(_label,_base,x,*args,**kwargs)
            setattr(model,name,types.MethodType(observed,model))
        return model

    def closed(engine):
        engine.model_runner.synchronize()
        write(ROOT/'raw'/cfg['name']/f'core_execution_enclosures_{os.getpid()}.json',dict(
            role=engine.stage_config.role,
            rows=[dict(metadata,ms=begin.elapsed_time(end)) for begin,end,metadata in samples],
            scope='Serial diagnostic CUDA Event enclosures. Recurrent graph replay includes output clones and host launch gaps. Graph input preparation, Prelude, and prompt Coda are outside the recurrent enclosure. These are not exclusive kernel unions or clean serving latency.'))
        return close(engine)
    ModelRunner._execute_batch=executed;RecurrentGraphs.execute=graphed
    AutoModelForCausalLM.from_pretrained=classmethod(loaded);StageEngine.close=closed
