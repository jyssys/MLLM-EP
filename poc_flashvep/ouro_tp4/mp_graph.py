"""Reuse exact-length stage math. Export through a persistent buffer, not a clone."""
import torch
from poc_flashvep.ouro_request_pp.graph_cache import GraphView


def graph_stage(part,hidden,rid,step,position,loop,start,stop):
    cache=part.caches[rid];prefill=step==0;length=hidden.shape[1] if prefill else position+1
    cache.length=length
    rkey=(prefill,length,position)
    if rkey not in part.rotary_constants:
        pos=torch.arange(length,device=part.device) if prefill else torch.tensor([position],device=part.device)
        part.rotary_constants[rkey]=part.rotary(hidden,pos.unsqueeze(0))
    part.prepared_rotary=part.rotary_constants[rkey]
    if loop not in cache.groups:
        cache.groups[loop]=(start,stop,torch.empty((stop-start,2,1,16,cache.capacity,128),dtype=hidden.dtype,device=part.device))
    _,_,pack=cache.groups[loop];key=('prefill' if prefill else 'exact',loop,length,start,stop)
    if key not in part.graphs:
        inp=torch.empty_like(hidden);inp.copy_(hidden)
        kv=torch.empty((stop-start,2,1,16,length,128),dtype=hidden.dtype,device=part.device)
        if not prefill:kv.copy_(pack[:,:,:,:,:length])
        pos=torch.arange(length,device=part.device) if prefill else torch.tensor([position],device=part.device)
        view=GraphView(kv,start,prefill=prefill)
        part.fixed_forward(inp,view,loop,start,stop,pos,None)
        torch.cuda.current_stream(part.device).synchronize()
        graph=torch.cuda.CUDAGraph();graph.capture_begin(capture_error_mode='thread_local')
        try:out=part.fixed_forward(inp,view,loop,start,stop,pos,None)
        finally:graph.capture_end()
        part.graphs[key]=(inp,kv,pos,out,graph)
    inp,kv,pos,out,graph=part.graphs[key]
    inp.copy_(hidden)
    if not prefill:kv.copy_(pack[:,:,:,:,:length])
    graph.replay();pack[:,:,:,:,:length].copy_(kv)
    # The caller copies this tensor to a request-private exported bank before reuse.
    return out
