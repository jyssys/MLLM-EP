"""Disjoint module boundaries and native-weight operator microbenchmarks."""
from collections import defaultdict
import time
import torch
import torch.nn.functional as F

from .common import append, write
from .probes import Controller
from .worker import timed

class Components:
    def __init__(self,a):
        self.a=a
        self.handles=[]
        self.events=[]
        self.pending={}

    def __enter__(self):
        for name,module in self.a.model.named_modules():
            kind=type(module).__name__
            category=None
            if kind=='CausalSoftmaxAttention':
                category='attention_rope_layout'
            elif name.endswith(('.wq','.wk','.wv','.wo')):
                category='attention_projections'
            elif kind=='TopKRouter':
                category='router'
            elif kind=='TransformerOutputLayer':
                category='output_norm_lm_head'
            elif kind in ('GroupRMSNorm','GroupLayerNorm') and not name.startswith('output.'):
                category='normalization'
            elif self.a.name=='foil1' and kind=='MultiheadSelfAttention':
                category='attention_with_projections_rope'
            elif self.a.name=='foil1' and kind=='Router':
                category='router'
            elif self.a.name=='foil1' and (name.endswith('ln_final') or name.endswith('lm_head')):
                category='output_norm_lm_head'
            elif self.a.name=='foil1' and kind=='RMSNorm' and not name.endswith('ln_final'):
                category='normalization'
            if category is None:
                continue
            def before(m,inputs,name=name,category=category):
                event=torch.cuda.Event(enable_timing=True)
                event.record()
                self.pending[name]=(event,category)
            def after(m,inputs,output,name=name):
                end=torch.cuda.Event(enable_timing=True)
                end.record()
                start,category=self.pending.pop(name)
                self.events.append((category,name,start,end))
            self.handles.append(module.register_forward_pre_hook(before))
            self.handles.append(module.register_forward_hook(after))
        return self

    def __exit__(self,*exc):
        for h in self.handles:
            h.remove()

def component_run(a,ids,target,length):
    # Expert and other categories do not contain each other.
    with Components(a) as components, Controller(a,'profile') as experts:
        torch.cuda.synchronize()
        start=time.perf_counter()
        a.forward(ids)
        torch.cuda.synchronize()
        wall=(time.perf_counter()-start)*1000
        events=[dict(category=cat,module=name,event_ms=s.elapsed_time(e)) for cat,name,s,e in components.events]
        events.extend(dict(category='expert_span',module=site,pass_index=r,event_ms=s.elapsed_time(e))
            for site,r,s,e in experts.times)
    write(target/f'components_p{length}.json',dict(model=a.name,wall_ms=wall,events=events,
        evidence='LIVE_CUDA_EVENT_MODULE_SPANS',
        limitation='Spans include internal launch gaps. Dispatch/combine outside these spans remains unresolved. Not a kernel service-time sum.'))

def microbench(a,ids,target,length,order_seed=None):
    with Controller(a,'probe') as probe:
        a.forward(ids)
        site=sorted(a.recurrent)[0]
        state=probe.router_state[site]
    ids_e=state['ids']
    order=ids_e.flatten().argsort(stable=True)
    x=state['x'][order//ids_e.shape[-1]].contiguous()
    groups=ids_e.flatten()[order]
    module=a.sites[site]
    e=getattr(module,'n_experts',getattr(module,'num_experts',None))
    def original(inp,controls):
        if a.name.startswith('ifm_'):
            return module.experts(inp,controls)
        offs=controls
        h1=F.grouped_mm(inp,module.experts_w1,offs=offs)
        h3=F.grouped_mm(inp,module.experts_w3,offs=offs)
        return F.grouped_mm(F.silu(h1)*h3,module.experts_w2,offs=offs)
    n=len(x)
    order_q=[0.,.05,.1,.2,.3,.5]
    if order_seed is not None:
        import random
        random.Random(order_seed).shuffle(order_q)
    for q in order_q:
        gen=torch.Generator(device=x.device).manual_seed(20261026)
        keep=torch.ones(n,dtype=torch.bool,device=x.device)
        keep[torch.randperm(n,device=x.device,generator=gen)[:int(n*q)]]=False
        compact=x[keep]
        counts=torch.bincount(groups[keep],minlength=e).tolist()
        controls=counts if a.name.startswith('ifm_') else torch.tensor(counts,dtype=torch.int32,device=x.device).cumsum(0).int()
        if q==0:
            expected=probe.previous[site]['outputs'].reshape(n,-1)[order]
            assert torch.equal(original(compact,controls),expected),'Native expert microbench does not match captured output'
        values=timed(lambda:original(compact,controls),20,5)
        append(target/'expert_microbench.jsonl',dict(model=a.name,site=site,input_length=length,
            target_q=q,effective_q=1-len(compact)/n,total_rows=n,executed_rows=len(compact),
            groups_nonempty=sum(c>0 for c in counts),groups=counts,wall_ms=values,
            method='V2_FOIL_GPU_OFFSETS_PRECOMPUTED_IFM_NATIVE_COUNTS_UNCHANGED',q0_native_output_parity=True,
            evidence='LIVE_NATIVE_WEIGHT_COMPONENT_MICROBENCH',
            excluded='Selection, compaction, cache, routing, attention, and combine. Not E2E speedup.'))

def generation_components(a,ids,target,cap=16):
    rows=[]
    outputs=[]
    with Controller(a,'profile') as profile:
        torch.cuda.synchronize()
        start=time.perf_counter()
        for step in range(cap):
            logits=a.forward(ids)
            token=logits[:,-1].argmax(-1,keepdim=True)
            torch.cuda.synchronize()
            rows.extend(dict(step=step,actual_prefix=ids.shape[-1],site=site,pass_index=r,event_ms=s.elapsed_time(e))
                for site,r,s,e in profile.times)
            outputs.append(token.item())
            ids=torch.cat([ids,token],dim=-1)
        torch.cuda.synchronize()
        wall=time.perf_counter()-start
    write(target/'generation_expert_events.json',dict(model=a.name,steps=cap,events=rows,
        observer_wall_s=wall,generated_ids=outputs,
        evidence='LIVE_EACH_FULL_PREFIX_FORWARD',
        limitation='Every step was measured. No last-step extrapolation. Heavy observer timing is not headline latency.'))

def micro_job(a,target,job):
    from .study import rows
    text=(rows(job)[0]['text']+'\n')*32
    for length in [128,512,2048]:
        microbench(a,a.encode(text,length),target,length,job['order_seed'])
    return dict(status='PASS',model=a.name,scope='Native-weight component microbench, not E2E')
