"""Nsight NVTX scopes without CUDA events at each small operator."""
import contextlib
import gc
import importlib
import os
import time
import torch
from poc_recurrent_asymmetry.contract import *
from poc_async_recurrent_pp.scripts.baseline import load


class Labels:
    def __init__(self,model):
        self.model=model;self.active=(-1,-1);self.restores=[];self.handles=[]
    def wrap(self,obj,name,kind):
        original=getattr(obj,name)
        def wrapped(*args,**kwargs):
            r,l=self.active
            with torch.cuda.nvtx.range(f'ASYM_COMP|{kind}|r={r}|l={l}'):
                return original(*args,**kwargs)
        self.restores.append((obj,name,original));setattr(obj,name,wrapped)
    def __enter__(self):
        for index,layer in enumerate(self.model.model.layers):
            def before(module,args,kwargs,index=index):self.active=(kwargs['current_ut'],index)
            self.handles.append(layer.register_forward_pre_hook(before,with_kwargs=True))
            for name in ['q_proj','k_proj','v_proj']:self.wrap(getattr(layer.self_attn,name),'forward','QKV')
            self.wrap(layer.self_attn.o_proj,'forward','O_PROJ');self.wrap(layer.mlp,'forward','MLP')
            for name in ['input_layernorm','input_layernorm_2','post_attention_layernorm','post_attention_layernorm_2']:
                self.wrap(getattr(layer,name),'forward','NORM')
        self.wrap(self.model.model.norm,'forward','PASS_NORM')
        self.wrap(self.model.model.rotary_emb,'forward','ROPE')
        self.wrap(self.model.model.early_exit_gate,'forward','GATE');self.wrap(self.model.lm_head,'forward','LM_HEAD')
        remote=importlib.import_module(type(self.model).__module__)
        self.wrap(remote.UniversalTransformerCache,'update','KV_UPDATE')
        self.wrap(remote,'apply_rotary_pos_emb','ROPE_APPLY')
        self.registry=remote.ALL_ATTENTION_FUNCTIONS;self.original=self.registry['sdpa']
        def attention(*args,**kwargs):
            r,l=self.active
            with torch.cuda.nvtx.range(f'ASYM_COMP|ATTN_CORE|r={r}|l={l}'):
                return self.original(*args,**kwargs)
        self.registry['sdpa']=attention
        return self
    def __exit__(self,*exc):
        for h in self.handles:h.remove()
        for obj,name,value in reversed(self.restores):setattr(obj,name,value)
        self.registry['sdpa']=self.original


@torch.inference_mode()
def main():
    output=Path(os.environ['LOOPED_KV_OUTPUT']);model,tokenizer=load()
    rows=read(ROOT/'metrics/frozen_workload.json')['requests']
    results=[]
    def request(P,repeat,labels):
        ids=torch.tensor([rows[0]['prompts'][str(P)]],device='cuda');token=torch.tensor([[13]],device='cuda')
        torch.cuda.synchronize();start=time.perf_counter()
        with torch.cuda.nvtx.range(f'ASYM_REQUEST|P={P}|rep={repeat}|labels={labels}'):
            with Labels(model) if labels else contextlib.nullcontext():
                with torch.cuda.nvtx.range('ASYM_PHASE|PREFILL'):
                    value=model(ids,use_cache=True,exit_at_step=3,logits_to_keep=1)
                logits=[value.logits[0,-1].clone()]
                for step in range(4):
                    with torch.cuda.nvtx.range(f'ASYM_PHASE|DECODE|step={step}'):
                        value=model(token,past_key_values=value.past_key_values,use_cache=True,exit_at_step=3,logits_to_keep=1)
                    logits.append(value.logits[0,-1].clone())
                torch.cuda.synchronize()
        wall=(time.perf_counter()-start)*1000
        result=dict(P=P,repeat=repeat,labels=labels,wall_ms=wall)
        del value,ids,token;gc.collect();torch.cuda.empty_cache()
        return result,logits
    reference={}
    for P in [1024,16384]:
        request(P,-1,False)
        for repeat in range(5):
            record,values=request(P,repeat,False);results.append(dict(lane='CLEAN',**record))
            if repeat==4:reference[P]=values
    torch.cuda.profiler.start()
    for P in [1024,16384]:
        for repeat in range(3):
            record,values=request(P,repeat,True)
            matched=sum(torch.equal(a,b) for a,b in zip(reference[P],values))
            results.append(dict(lane='NSIGHT_NVTX',**record,logit_checkpoints=5,bitwise_logit_matches=matched))
            assert matched==5,'Read-only NVTX observer changed exact logits'
    torch.cuda.profiler.stop()
    write(output/'timing_and_parity.json',results)
    write(output/'status.json',dict(success=True,profiled_requests=6,decode_steps_per_request=4,
        context_lengths=[1024,16384],physical_gpu=int(os.environ['CUDA_VISIBLE_DEVICES']),
        scope='Nsight diagnostic timeline. Separate clean repeats. No last-forward multiplication.'))


if __name__=='__main__':main()
