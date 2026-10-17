"""One component family per run to measure observer cost explicitly."""
import gc
import importlib
import os
import statistics
import time
import torch
from poc_recurrent_asymmetry.contract import *
from poc_async_recurrent_pp.scripts.baseline import load


class Component:
    def __init__(self,model,kind):self.model=model;self.kind=kind;self.restores=[];self.handles=[];self.events=[];self.active=(-1,-1)
    def wrap(self,obj,name):
        original=getattr(obj,name)
        def measured(*args,**kwargs):
            a=torch.cuda.Event(enable_timing=True);b=torch.cuda.Event(enable_timing=True)
            a.record();r,l=self.active
            try:return original(*args,**kwargs)
            finally:b.record();self.events.append((r,l,a,b))
        self.restores.append((obj,name,original));setattr(obj,name,measured)
    def __enter__(self):
        for i,layer in enumerate(self.model.model.layers):
            def active(module,args,kwargs,i=i):self.active=(kwargs['current_ut'],i)
            self.handles.append(layer.register_forward_pre_hook(active,with_kwargs=True))
            if self.kind=='MLP':self.wrap(layer.mlp,'forward')
            if self.kind=='O_PROJ':self.wrap(layer.self_attn.o_proj,'forward')
        remote=importlib.import_module(type(self.model).__module__)
        if self.kind=='KV_UPDATE':self.wrap(remote.UniversalTransformerCache,'update')
        if self.kind=='ATTN_CORE':
            registry=remote.ALL_ATTENTION_FUNCTIONS;original=registry['sdpa']
            def attention(*args,**kwargs):
                a=torch.cuda.Event(enable_timing=True);b=torch.cuda.Event(enable_timing=True);a.record();r,l=self.active
                try:return original(*args,**kwargs)
                finally:b.record();self.events.append((r,l,a,b))
            registry['sdpa']=attention;self.registry=(registry,original)
        return self
    def __exit__(self,*exc):
        for h in self.handles:h.remove()
        for obj,name,old in reversed(self.restores):setattr(obj,name,old)
        if self.kind=='ATTN_CORE':self.registry[0]['sdpa']=self.registry[1]
    def finish(self):
        torch.cuda.synchronize()
        return [dict(round=r,layer=l,duration_ms=a.elapsed_time(b)) for r,l,a,b in self.events]


@torch.inference_mode()
def main():
    output=Path(os.environ['LOOPED_KV_OUTPUT']);model,tokenizer=load()
    frozen=read(ROOT/'metrics/frozen_workload.json')['requests'][0];rows=[]
    def case(P,phase,kind):
        ids=torch.tensor([frozen['prompts'][str(P)]],device='cuda');token=torch.tensor([[13]],device='cuda')
        value=model(ids,use_cache=True,exit_at_step=3,logits_to_keep=1) if phase=='decode' else None
        gc.collect();torch.cuda.synchronize()
        observer=Component(model,kind) if kind else None
        if observer:observer.__enter__()
        start=time.perf_counter()
        if phase=='prefill':value=model(ids,use_cache=True,exit_at_step=3,logits_to_keep=1)
        else:
            for _ in range(8):value=model(token,past_key_values=value.past_key_values,use_cache=True,exit_at_step=3,logits_to_keep=1)
        torch.cuda.synchronize();wall=(time.perf_counter()-start)*1000
        if observer:observer.__exit__()
        events=observer.finish() if observer else []
        logits=value.logits.detach().clone();del value,ids,token;gc.collect();torch.cuda.empty_cache()
        return wall,events,logits
    for P in [1024,16384]:
        for phase in ['prefill','decode']:
            for _ in range(3):case(P,phase,None)
            clean=[case(P,phase,None) for _ in range(5)]
            times=[r[0] for r in clean];reference=clean[-1][2]
            for kind in ['ATTN_CORE','KV_UPDATE','MLP','O_PROJ']:
                for repeat in range(5):
                    wall,events,logits=case(P,phase,kind)
                    assert torch.equal(logits,reference),'Component observer changed exact logits'
                    expected=96*(8 if phase=='decode' else 1)
                    assert len(events)==expected
                    rows.append(dict(P=P,phase=phase,component=kind,repeat=repeat,wall_ms=wall,
                        clean_wall_ms=times,observer_percent=(wall/statistics.median(times)-1)*100,
                        component_sum_ms=sum(r['duration_ms'] for r in events),events=events,
                        bitwise_logits=True,cached_steps=8 if phase=='decode' else 0,
                        scope='Single component family per run. Do not sum different runs into an exact E2E partition.'))
            write(output/f'P{P}_{phase}.json',[r for r in rows if r['P']==P and r['phase']==phase])
            print('LIGHT',P,phase,flush=True)
    write(output/'status.json',dict(success=True,quality='Read-only observer logits parity.',
        scope='One overlapping calibration request. Technical repeats, not independent benchmark requests.'))


if __name__=='__main__':main()
