"""Last-token component observations. Never use this observer's wall time as serving latency."""
import torch

class Trace:
    def __init__(self,model,native=False,rank=0):
        self.frames={};self.step=0;self.loop=0;self.hooks=[];self.rank=rank;self.native=native
        for layer_index,layer in enumerate(model.model.layers):
            def pre(m,args,kwargs):self.loop=int(args[2] if self.native else kwargs['current_ut'])
            self.hooks.append(layer.register_forward_pre_hook(pre,with_kwargs=True))
            def capture(op,index):
                def hook(m,args,value):
                    if self.rank:return
                    if isinstance(value,tuple):value=(value[0]+value[1] if op=='layer' and self.native else value[0])
                    x=value.reshape(-1,value.shape[-1])[-1:].detach().float().cpu()
                    self.frames[f'{self.step}:{self.loop}:{index}:{op}']=x
                return hook
            self.hooks.extend([layer.self_attn.register_forward_hook(capture('attention',layer_index)),
                layer.mlp.register_forward_hook(capture('MLP',layer_index)),layer.register_forward_hook(capture('layer',layer_index))])
    def close(self):
        for h in self.hooks:h.remove()

def errors(a,b):
    a=a.float();b=b.float();d=a-b
    return dict(max_abs=float(d.abs().max()),MAE=float(d.abs().mean()),relative_L2=float(d.norm()/b.norm().clamp_min(1e-20)))
