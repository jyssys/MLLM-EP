"""Exact-length graph scratch cache. No masked future KV enters SDPA.

The common runtime stages owner-local KV to an exact-length graph buffer.
This adds local GPU copies in both arms. It is not a new attention algorithm.
"""
import torch


class ExactCache:
    def __init__(self,capacity,remote):
        self.capacity=capacity
        self.original=remote.UniversalTransformerCache(96)
        self.groups={}
        self.length=0

    def reset(self,remote):
        self.original=remote.UniversalTransformerCache(96)
        self.length=0

    def pack_prefill(self,loop,start,stop):
        length=self.original.key_cache[24*loop+start].shape[2]
        self.length=length
        if loop not in self.groups:
            k=self.original.key_cache[24*loop+start]
            self.groups[loop]=(start,stop,torch.empty((stop-start,2,1,16,self.capacity,128),dtype=k.dtype,device=k.device))
        _,_,pack=self.groups[loop]
        for j,i in enumerate(range(start,stop)):
            pack[j,0,:,:,:length].copy_(self.original.key_cache[24*loop+i])
            pack[j,1,:,:,:length].copy_(self.original.value_cache[24*loop+i])
        # Original prefill tensors are no longer needed by another layer or rank.
        for i in range(start,stop):
            self.original.key_cache[24*loop+i]=None
            self.original.value_cache[24*loop+i]=None

    def summary(self):
        slots=[24*r+i for r,(start,stop,_) in self.groups.items() for i in range(start,stop)]
        return dict(slots=sorted(slots),shapes={s:[1,16,self.length,128] for s in slots},
                    bytes=sum(pack.numel()*pack.element_size() for _,_,pack in self.groups.values()),
                    distinct_storage=True,allocation='Disjoint slices of owner-local packed storage, not separate allocations.')

    def snapshot(self):
        return {24*r+i:(pack[j,0,:,:,:self.length].cpu(),pack[j,1,:,:,:self.length].cpu())
                for r,(start,stop,pack) in self.groups.items() for j,i in enumerate(range(start,stop))}


class GraphView:
    def __init__(self,kv,start,prefill=False):
        self.kv=kv;self.start=start;self.prefill=prefill

    def update(self,k,v,index,metadata):
        j=index%24-self.start
        if self.prefill:
            self.kv[j,0].copy_(k);self.kv[j,1].copy_(v)
            # Preserve the original prefill K/V strides and SDPA dispatch.
            return k,v
        self.kv[j,0,:,:,-1:].copy_(k);self.kv[j,1,:,:,-1:].copy_(v)
        return self.kv[j,0],self.kv[j,1]


def exact_graph_stage(part,hidden,rid,step,position,loop,start,stop,lock):
    cache=part.caches[rid]
    if step==0 and not part.capture_enabled:
        pos=torch.arange(hidden.shape[1],device=part.device)
        out=part.fixed_forward(hidden,cache.original,loop,start,stop,pos,None)
        cache.pack_prefill(loop,start,stop)
        return out
    prefill=step==0
    length=hidden.shape[1] if prefill else position+1
    if part.precompute_rotary:
        rotary_key=(prefill,length,position)
        if rotary_key not in part.rotary_constants:
            positions=(torch.arange(length,device=part.device) if prefill else
                       torch.tensor([position],dtype=torch.long,device=part.device)).unsqueeze(0)
            part.rotary_constants[rotary_key]=part.rotary(hidden,positions)
        part.prepared_rotary=part.rotary_constants[rotary_key]
    cache.length=length
    if loop not in cache.groups:
        cache.groups[loop]=(start,stop,torch.empty((stop-start,2,1,16,cache.capacity,128),
                                                dtype=hidden.dtype,device=hidden.device))
    _,_,pack=cache.groups[loop]
    if not part.capture_enabled:
        kv=pack[:,:,:,:,:length].contiguous()
        pos=torch.tensor([position],dtype=torch.long,device=part.device)
        out=part.fixed_forward(hidden,GraphView(kv,start),loop,start,stop,pos,None)
        pack[:,:,:,:,:length].copy_(kv)
        return out
    key=('prefill' if prefill else 'exact',loop,length,start,stop)
    if key not in part.graphs:
        inp=torch.empty_like(hidden)
        kv=torch.empty((stop-start,2,1,16,length,128),dtype=hidden.dtype,device=hidden.device)
        pos=torch.arange(length,device=part.device) if prefill else torch.tensor([position],dtype=torch.long,device=part.device)
        view=GraphView(kv,start,prefill=prefill)
        inp.copy_(hidden)
        if not prefill:kv.copy_(pack[:,:,:,:,:length])
        with lock:
            part.fixed_forward(inp,view,loop,start,stop,pos,None)
            torch.cuda.current_stream(part.device).synchronize()
            graph=torch.cuda.CUDAGraph()
            # Direct capture avoids per-graph global synchronization and Python GC.
            # All setup remains outside measured requests.
            graph.capture_begin(capture_error_mode='thread_local')
            try:
                out=part.fixed_forward(inp,view,loop,start,stop,pos,None)
            finally:
                graph.capture_end()
        # Inputs allocated outside capture need explicit lifetime ownership.
        # Replays read the position tensor even when no Python function uses it.
        part.graphs[key]=(inp,kv,pos,out,graph)
    inp,kv,pos,out,graph=part.graphs[key]
    inp.copy_(hidden)
    if not prefill:kv.copy_(pack[:,:,:,:,:length])
    graph.replay()
    pack[:,:,:,:,:length].copy_(kv)
    # Graph scratch is shared only by this rank's serial worker.
    # Publish an independent tensor before the next request reuses that scratch.
    return out.clone()
