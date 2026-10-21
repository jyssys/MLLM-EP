"""Owner-local exact KV with bounded graph storage and the original Ouro gate."""
import torch
from poc_flashvep.ouro_request_pp.model_adapter import Part,StaticCache

class Stage(Part):
    def __init__(self,rank,placement):
        super().__init__(rank,placement);self.local_graphs={}

    def prepare(self,requests,cap,owned):
        self.caches={r['id']:StaticCache(len(r['input_ids'])+cap-1) for r in requests}
        for r in requests:
            P=len(r['input_ids']);h=torch.zeros((1,P,2048),device=self.device,dtype=torch.bfloat16)
            for loop,first,stop in owned:
                self.execute(h,r['id'],0,0,loop,first,stop)
                self.execute(h[:,:1],r['id'],1,P,loop,first,stop)
        torch.cuda.synchronize(self.device)
        # Preparation only populates scratch. Each real prefill overwrites all historical prefix positions.

    def block(self,h,cache,loop,first,stop,pos,mask):
        rotary=self.rotary(h,pos.unsqueeze(0))
        for index in range(first,stop):
            h=self.layers[str(index)](h,attention_mask=mask,position_ids=pos.unsqueeze(0),past_key_value=cache,
                cache_position=pos,position_embeddings=rotary,current_ut=loop,use_cache=True)
        gate=None
        if stop==24:h=self.norm(h);gate=self.gate(h[:,-1:,:])
        return h,gate

    def execute(self,h,rid,step,position,loop,first,stop):
        cache=self.caches[rid];cache.prefill=step==0;cache.length=position+h.shape[1]
        if step==0:
            pos=torch.arange(h.shape[1],device=self.device)
            return self.block(h,cache,loop,first,stop,pos,None)
        if not self.capture_enabled:
            pos=torch.tensor([position],device=self.device)
            mask=torch.where(torch.arange(cache.capacity,device=self.device)<=pos[0],0.,float('-inf')).to(torch.bfloat16).view(1,1,1,-1)
            return self.block(h,cache,loop,first,stop,pos,mask)
        key=(rid,loop,first,stop)
        if key not in self.local_graphs:
            inp=torch.empty_like(h);inp.copy_(h);pos=torch.tensor([position],device=self.device)
            full=torch.arange(cache.capacity,device=self.device)
            def call():
                mask=torch.where(full<=pos[0],0.,float('-inf')).to(torch.bfloat16).view(1,1,1,-1)
                return self.block(inp,cache,loop,first,stop,pos,mask)
            call();torch.cuda.synchronize(self.device);graph=torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):out,gate=call()
            # CUDA Graph does not retain tensors allocated before capture.
            # The mask kernel reads this arange buffer on every later replay.
            self.local_graphs[key]=(inp,pos,out,gate,graph,full)
        inp,pos,out,gate,graph,full=self.local_graphs[key];inp.copy_(h);pos.fill_(position);graph.replay()
        return out,gate

    def export_exit(self,rid,loop,position,bank):
        cache=self.caches[rid]
        for j,index in enumerate(self.indices):
            slot=24*loop+index
            bank[j,0].copy_(cache.key_cache[slot][:,:,position,:])
            bank[j,1].copy_(cache.value_cache[slot][:,:,position,:])

    def import_exit(self,rid,source_loop,target_loops,position,bank):
        cache=self.caches[rid]
        for loop in target_loops:
            for j,index in enumerate(self.indices):
                slot=24*loop+index
                cache.key_cache[slot][:,:,position,:].copy_(bank[j,0])
                cache.value_cache[slot][:,:,position,:].copy_(bank[j,1])
        # A deeper owner can receive valid KV without executing that loop.
        # Its exported snapshot must include the newly published current row.
        cache.length=max(cache.length,position+1)

    def fill_local(self,rid,loop,position):
        cache=self.caches[rid]
        for later in range(loop+1,4):
            for index in self.indices:
                src,dst=24*loop+index,24*later+index
                cache.key_cache[dst][:,:,position,:].copy_(cache.key_cache[src][:,:,position,:])
                cache.value_cache[dst][:,:,position,:].copy_(cache.value_cache[src][:,:,position,:])

def decision(signal,loop,cdf,survival):
    hazard=float(torch.sigmoid(signal.reshape(-1)[0]).float())
    cdf+=hazard*survival;survival*=1-hazard
    return (loop>=1 and cdf>=.5) or loop==3,cdf,survival,hazard
