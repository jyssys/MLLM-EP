"""Observe real source layer frontiers with asynchronous CUDA events."""
import atexit,time
from .common import *


def install(cfg,role):
    if role!='prefill' or not cfg.get('boundary_events'):return
    import torch,vllm
    from vllm.model_executor.models.qwen3 import Qwen3DecoderLayer,Qwen3ForCausalLM
    from vllm.v1.worker.gpu.model_runner import GPUModelRunner
    from vllm.v1.engine.input_processor import InputProcessor
    state={};identities={};events=[]
    assigned=InputProcessor.assign_request_id
    def assign(request):
        result=assigned(request);identities[request.request_id]=request.external_req_id;return result
    InputProcessor.assign_request_id=staticmethod(assign)
    prepare=GPUModelRunner.prepare_inputs
    def prepared(self,*args,**kwargs):
        batch=prepare(self,*args,**kwargs);state['batch']=batch;return batch
    GPUModelRunner.prepare_inputs=prepared
    forward=Qwen3DecoderLayer.forward
    def layer(self,*args,**kwargs):
        result=forward(self,*args,**kwargs)
        batch=state.get('batch');number=int(self.self_attn.attn.layer_name.split('.')[2])+1
        if batch is not None and number in (24,28,32,34,36):
            names=[identities.get(rid,rid) for rid in batch.req_ids]
            if names and all(r.startswith('q') and r[1:].isdigit() for r in names):
                assert all(bool(x) for x in batch.is_prefilling_np[:batch.num_reqs])
                event=torch.cuda.Event(enable_timing=True);event.record()
                events.append((event,dict(requests=names,completed_layer=number,query_tokens=batch.num_tokens,host_submit=time.monotonic())))
        return result
    Qwen3DecoderLayer.forward=layer
    outer=Qwen3ForCausalLM.forward
    def model(self,*args,**kwargs):
        try:return outer(self,*args,**kwargs)
        finally:state.pop('batch',None)
    Qwen3ForCausalLM.forward=model
    llm=vllm.LLM
    def initialized(*args,**kwargs):
        value=llm(*args,**kwargs);torch.cuda.synchronize()
        anchor=torch.cuda.Event(enable_timing=True);lo=time.monotonic();anchor.record();anchor.synchronize();hi=time.monotonic()
        state.update(anchor=anchor,host_lower=lo,host_upper=hi);return value
    vllm.LLM=initialized
    def export():
        torch.cuda.synchronize();anchor=state['anchor'];mid=(state['host_lower']+state['host_upper'])/2
        write(ROOT/'raw'/cfg['name']/'source_layer_boundary_events.json',dict(
            anchor_host_lower=state['host_lower'],anchor_host_upper=state['host_upper'],
            rows=[dict(meta,time=mid+anchor.elapsed_time(event)/1000) for event,meta in events],
            scope='Actual source CUDA stream-frontier events. They are not exclusive kernel times. Instrumentation qualification is separate.'))
    atexit.register(export)
