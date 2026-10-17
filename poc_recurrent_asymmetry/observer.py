"""GPU-stream intervals. No arithmetic changes or per-component host sync."""
import importlib
import torch


class Observer:
    def __init__(self,model,detail=True):
        self.model=model;self.detail=detail;self.handles=[];self.restores=[]
        self.records=[];self.active=(None,None);self.chunk=0

    def event(self):
        event=torch.cuda.Event(enable_timing=True);event.record();return event

    def wrap(self,obj,name,category):
        original=getattr(obj,name)
        def wrapped(*args,**kwargs):
            start=self.event();r,l=self.active;c=self.chunk
            try:return original(*args,**kwargs)
            finally:self.records.append((category,r,l,c,start,self.event()))
        self.restores.append((obj,name,original));setattr(obj,name,wrapped)

    def __enter__(self):
        self.origin=self.event()
        for index,layer in enumerate(self.model.model.layers):
            def enter(module,args,kwargs,index=index):
                self.active=(kwargs['current_ut'],index)
                module._asymmetry_start=self.event()
            def leave(module,args,value,index=index):
                self.records.append(('LAYER',self.active[0],index,self.chunk,module._asymmetry_start,self.event()))
            self.handles.append(layer.register_forward_pre_hook(enter,with_kwargs=True))
            self.handles.append(layer.register_forward_hook(leave))
            if self.detail:
                for name in ['q_proj','k_proj','v_proj']:
                    self.wrap(getattr(layer.self_attn,name),'forward','QKV')
                self.wrap(layer.self_attn.o_proj,'forward','O_PROJ')
                self.wrap(layer.mlp,'forward','MLP')
                for name in ['input_layernorm','input_layernorm_2','post_attention_layernorm','post_attention_layernorm_2']:
                    self.wrap(getattr(layer,name),'forward','NORM')
        self.wrap(self.model.model.norm,'forward','PASS_NORM')
        if self.detail:
            self.wrap(self.model.model.rotary_emb,'forward','ROPE')
            self.wrap(self.model.model.early_exit_gate,'forward','GATE')
            self.wrap(self.model.lm_head,'forward','LM_HEAD')
            module=importlib.import_module(type(self.model).__module__)
            self.wrap(module.UniversalTransformerCache,'update','KV_UPDATE')
            registry=module.ALL_ATTENTION_FUNCTIONS
            original=registry['sdpa']
            def attention(*args,**kwargs):
                start=self.event();r,l=self.active;c=self.chunk
                try:return original(*args,**kwargs)
                finally:self.records.append(('ATTN_CORE',r,l,c,start,self.event()))
            self.registry_original=(registry,original)
            registry['sdpa']=attention
        return self

    def __exit__(self,*exc):
        for h in self.handles:h.remove()
        for obj,name,value in reversed(self.restores):setattr(obj,name,value)
        if self.detail:self.registry_original[0]['sdpa']=self.registry_original[1]

    def resolve(self):
        torch.cuda.synchronize()
        return [dict(category=category,round=r,layer=l,chunk=c,
            start_ms=self.origin.elapsed_time(a),end_ms=self.origin.elapsed_time(b),duration_ms=a.elapsed_time(b),
            scope='Single-GPU stream interval. Includes enqueue gaps. LAYER contains child components.')
            for category,r,l,c,a,b in self.records]
