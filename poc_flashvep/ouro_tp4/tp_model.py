"""Shard official HF linear layers. Keep native decoder math and KV indices."""
import gc
import importlib
import torch
import torch.distributed as dist
from torch import nn
from poc_flashvep.ouro_request_pp.model_adapter import load_cpu,nbytes
from .contract import UUIDS


class CollectiveMeter:
    def __init__(self):self.operations=[]
    def record(self,kind,x):
        self.operations.append(dict(kind=kind,shape=list(x.shape),dtype=str(x.dtype),logical_tensor_bytes=x.numel()*x.element_size()))


class RowParallel(nn.Linear):
    def __init__(self,source,rank,meter):
        width=source.in_features//4
        super().__init__(width,source.out_features,bias=False,dtype=source.weight.dtype)
        self.weight=nn.Parameter(source.weight[:,rank*width:(rank+1)*width].clone(),requires_grad=False)
        self.meter=meter
    def forward(self,x):
        # Native BF16 GEMM accumulates in FP32 before its final BF16 cast.
        # Keep the distributed row partials in FP32 until that same final cast.
        y=torch.mm(x.reshape(-1,self.in_features),self.weight.t(),out_dtype=torch.float32).view(*x.shape[:-1],self.out_features)
        self.meter.record('all_reduce',y)
        dist.all_reduce(y,op=dist.ReduceOp.SUM)
        return y.to(x.dtype)


class ColumnParallel(nn.Linear):
    def forward(self,x):
        y=torch.mm(x.reshape(-1,self.in_features),self.weight.t(),out_dtype=torch.float32)
        return y.to(x.dtype).view(*x.shape[:-1],self.out_features)


def column(source,rank):
    width=source.out_features//4
    out=ColumnParallel(source.in_features,width,bias=False,dtype=source.weight.dtype)
    out.weight=nn.Parameter(source.weight[rank*width:(rank+1)*width].clone(),requires_grad=False)
    return out


class KVView:
    def __init__(self,kv,prefill):self.kv=kv;self.prefill=prefill
    def update(self,k,v,index,metadata):
        if self.prefill:
            self.kv[index,0].copy_(k);self.kv[index,1].copy_(v)
            return k,v
        self.kv[index,0,:,:,-1:].copy_(k);self.kv[index,1,:,:,-1:].copy_(v)
        return self.kv[index,0],self.kv[index,1]


class TPModel:
    def __init__(self,rank,attention_geometry_control=False):
        self.rank=rank;self.device=torch.device('cuda',rank);self.meter=CollectiveMeter()
        model=load_cpu()
        self.remote=importlib.import_module(type(model).__module__)
        self.config=model.config
        if attention_geometry_control:
            original=self.remote.ALL_ATTENTION_FUNCTIONS['sdpa']
            def geometry(module,q,k,v,attention_mask,**kwargs):
                # Diagnostic: four unique local heads, repeated to the native 16-head launch geometry.
                B,H,L,D=q.shape
                def repeat_heads(x,transpose_layout):
                    if transpose_layout:
                        out=torch.empty((B,x.shape[2],16,D),device=x.device,dtype=x.dtype).transpose(1,2)
                    else:out=torch.empty((B,16,x.shape[2],D),device=x.device,dtype=x.dtype)
                    for j in range(4):out[:,4*j:4*j+4].copy_(x)
                    return out
                out,weights=original(module,repeat_heads(q,True),repeat_heads(k,L>1),repeat_heads(v,L>1),attention_mask,**kwargs)
                return out[:,:,:4,:],weights
            self.remote.ALL_ATTENTION_FUNCTIONS.register('tp_geometry_control',geometry)
            self.config._attn_implementation='tp_geometry_control'
        assert self.config.num_attention_heads==self.config.num_key_value_heads==16
        assert self.config.intermediate_size==5632 and self.config.hidden_size==2048
        for layer in model.model.layers:
            a=layer.self_attn
            for name in ['q_proj','k_proj','v_proj']:setattr(a,name,column(getattr(a,name),rank))
            a.o_proj=RowParallel(a.o_proj,rank,self.meter)
            layer.mlp.gate_proj=column(layer.mlp.gate_proj,rank)
            layer.mlp.up_proj=column(layer.mlp.up_proj,rank)
            layer.mlp.down_proj=RowParallel(layer.mlp.down_proj,rank,self.meter)
        self.layers=model.model.layers.to(self.device)
        self.embedding=model.model.embed_tokens.to(self.device)
        self.norm=model.model.norm.to(self.device)
        self.gate=model.model.early_exit_gate.to(self.device)
        self.head=model.lm_head.to(self.device)
        self.rotary=model.model.rotary_emb.to(self.device)
        self.weights=dict(core=nbytes(self.layers),embedding=nbytes(self.embedding),head=nbytes(self.head),norm=nbytes(self.norm),gate=nbytes(self.gate))
        self.weights['total']=sum(self.weights.values())
        self.graphs={};self.caches={};self.capacity=0;self.lengths={}
        del model;gc.collect()
        assert str(torch.cuda.get_device_properties(rank).uuid).removeprefix('GPU-')==UUIDS[rank]

    def configure(self,requests,O):
        self.caches={r['id']:torch.empty((96,2,1,4,len(r['input_ids'])+O-1,128),
            device=self.device,dtype=torch.bfloat16) for r in requests}
        self.lengths={r['id']:0 for r in requests}

    def core(self,hidden,kv,pos,rotary):
        positions=pos.unsqueeze(0)
        for loop in range(4):
            for layer in self.layers:
                hidden=layer(hidden,attention_mask=None,position_ids=positions,past_key_value=kv,use_cache=True,
                    cache_position=pos,position_embeddings=rotary,current_ut=loop)
            hidden=self.norm(hidden);self.gate(hidden)
        return hidden

    def forward(self,ids,rid,step,P,graphs=True):
        hidden=self.embedding(ids)
        length=P if step==0 else P+step
        self.lengths[rid]=length
        if not graphs:
            pos=torch.arange(P,device=self.device) if step==0 else torch.tensor([length-1],device=self.device)
            # Exact-length scratch gives the same SDPA key length, never masked future KV.
            kv=torch.empty((96,2,1,4,length,128),device=self.device,dtype=torch.bfloat16)
            if step:kv.copy_(self.caches[rid][:,:,:,:,:length])
            out=self.core(hidden,KVView(kv,step==0),pos,self.rotary(hidden,pos.unsqueeze(0)))
            self.caches[rid][:,:,:,:,:length].copy_(kv)
        else:
            key=(step==0,length)
            if key not in self.graphs:
                pos=torch.arange(P,device=self.device) if step==0 else torch.tensor([length-1],device=self.device)
                inp=torch.empty_like(hidden);inp.copy_(hidden)
                kv=torch.empty((96,2,1,4,length,128),device=self.device,dtype=torch.bfloat16)
                if step:kv.copy_(self.caches[rid][:,:,:,:,:length])
                rotary=self.rotary(hidden,pos.unsqueeze(0));view=KVView(kv,step==0)
                self.core(inp,view,pos,rotary)
                torch.cuda.synchronize(self.device)
                dist.barrier()
                before=len(self.meter.operations)
                graph=torch.cuda.CUDAGraph()
                graph.capture_begin(capture_error_mode='thread_local')
                try:out=self.core(inp,view,pos,rotary)
                finally:graph.capture_end()
                ops=self.meter.operations[before:]
                assert len(ops)==192
                self.graphs[key]=(inp,kv,pos,rotary,out,graph,ops)
            inp,kv,pos,rotary,out,graph,ops=self.graphs[key]
            inp.copy_(hidden)
            if step:kv.copy_(self.caches[rid][:,:,:,:,:length])
            graph.replay()
            self.caches[rid][:,:,:,:,:length].copy_(kv)
        logits=self.head(out[:,-1:,:])
        token=logits[:,-1,:].argmax(-1).view(1,1)
        self.meter.record('broadcast',token)
        dist.broadcast(token,src=0)
        return logits,token,out[:,-1:,:].clone()

    def memory(self):
        return dict(rank=self.rank,physical_GPU=self.rank+4,weights=self.weights,
            logical_KV_bytes=sum(96*2*4*L*128*2 for L in self.lengths.values()),
            reserved_KV_bytes=sum(x.numel()*x.element_size() for x in self.caches.values()),
            graph_scratch_bytes=sum(kv.numel()*kv.element_size()+inp.numel()*inp.element_size()
                for inp,kv,_,_,_,_,_ in self.graphs.values()),
            allocated_bytes=torch.cuda.memory_allocated(self.device),reserved_bytes=torch.cuda.memory_reserved(self.device),
            peak_bytes=torch.cuda.max_memory_allocated(self.device),head_range=[4*self.rank,4*self.rank+4],
            KV_slots=96,head_count=4)
