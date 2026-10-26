"""Bounded P1 observer: independent native requests, NO expert coalescing.

Native model threads stop only at the routed MoE boundary. A single coordinator
orders all EP collectives. CUDA events preserve payload and output dependencies.
This diagnostic is not an optimized serving engine or an A implementation.
"""
from collections import Counter
from dataclasses import dataclass
import queue
import threading
import time
import torch
import torch.distributed as dist
from .common import append,write
from .measure import shaped_inputs
from .worker import greedy

@dataclass(eq=False)
class Work:
    request:int
    site:str
    loop:int
    step:int
    hidden:object
    ids:object
    weights:object
    ready:object
    result:object=None
    complete:object=None
    observed_ready:float=None
    counts:object=None
    signal:object=None

class Front:
    def __init__(self):
        self.incoming=queue.Queue()
        self.local=threading.local()
    def run(self,site,hidden,ids,weights):
        ctx=self.local.context
        loop=ctx['calls'][site]
        ctx['calls'][site]+=1
        ready=torch.cuda.Event()
        ready.record()
        work=Work(ctx['request'],site,loop,ctx['step'],hidden,ids,weights,ready,signal=threading.Event())
        self.incoming.put(work)
        assert work.signal.wait(180),'Ready observer coordinator timed out'
        stream=torch.cuda.current_stream()
        stream.wait_event(work.complete)
        for value in work.result:
            if isinstance(value,torch.Tensor):value.record_stream(stream)
        return work.result

def observe(a,ep,target,config):
    cap=config.get('probe_tokens',8)
    cases=[]
    for b in [1,2,4]:
        for pattern in ['synchronized','staggered']:
            prefixes=[shaped_inputs(a,1,128,i)[0] for i in range(b)]
            expected=[greedy(a,ep,ids,cap) for ids in prefixes]
            front=Front()
            for block in a.sites.values():block._ep=front
            outputs={};errors=[];pending=[];observations=[];threads=[]
            origin=time.perf_counter()+.1
            parent_stream=torch.cuda.current_stream()
            arrivals=[0. if pattern=='synchronized' else i*.025 for i in range(b)]
            def request_thread(r):
                try:
                    with torch.inference_mode():
                        stream=torch.cuda.Stream()
                        stream.wait_stream(parent_stream)
                        with torch.cuda.stream(stream):
                            prefixes[r].record_stream(stream)
                            while time.perf_counter()<origin+arrivals[r]:time.sleep(.0001)
                            ids=prefixes[r].clone()
                            generated=[]
                            for step in range(cap):
                                front.local.context=dict(request=r,step=step,calls=Counter())
                                logits=a.forward(ids)
                                expected_calls={site:a.loops if site in a.recurrent else 1 for site in a.sites}
                                assert dict(front.local.context['calls'])==expected_calls
                                token=logits[:,-1].argmax(-1,keepdim=True)
                                generated.append(token.item())
                                ids=torch.cat([ids,token],dim=-1)
                            stream.synchronize()
                            outputs[r]=generated
                except BaseException as error:errors.append(repr(error))
            for r in range(b):
                thread=threading.Thread(target=request_thread,args=(r,),daemon=True)
                thread.start();threads.append(thread)
            sites=sorted(a.sites)
            until=time.monotonic()+config.get('probe_timeout_s',600)
            try:
                while True:
                    while True:
                        try:pending.append(front.incoming.get_nowait())
                        except queue.Empty:break
                    assert not errors,errors
                    assert time.monotonic()<until,'Bounded ready-observer timeout'
                    if ep.rank==0:
                        ready=[]
                        for work in pending:
                            if work.ready.query():
                                if work.observed_ready is None:work.observed_ready=time.perf_counter()
                                if work.counts is None:
                                    block=a.sites[work.site]
                                    experts=block.n_experts if a.name.startswith('ifm') else block.num_experts
                                    work.counts=torch.bincount(work.ids.flatten(),minlength=experts).cpu().tolist()
                                ready.append(work)
                        if ready:
                            work=min(ready,key=lambda w:w.observed_ready)
                            command=[1,work.request,sites.index(work.site),work.loop,work.step]
                            observations.append(dict(batch=b,arrival=pattern,chosen=command[1:],
                                ready=[dict(request=w.request,site=w.site,loop=w.loop,step=w.step,
                                    ready_host_s=w.observed_ready-origin,expert_counts=w.counts) for w in ready],
                                chosen_host_s=time.perf_counter()-origin,
                                evidence='LIVE_DEVICE_READY_EVENTS_FROM_INDEPENDENT_NATIVE_REQUESTS',
                                coalescing_enabled=False,poll_resolution_s=.0001))
                        elif len(outputs)==b:
                            command=[0,0,0,0,0]
                        else:
                            time.sleep(.0001)
                            continue
                        candidates=[(w.request,sites.index(w.site),w.loop,w.step) for w in ready] if command[0] else []
                        flat_candidates=[v for item in candidates for v in item]
                        payload=command+flat_candidates+[-1]*(4*b-len(flat_candidates))
                        header=torch.tensor(payload,device='cuda',dtype=torch.int64)
                    else:header=torch.empty(5+4*b,device='cuda',dtype=torch.int64)
                    dist.broadcast(header,src=0)
                    payload=header.tolist()
                    command=payload[:5]
                    if command[0]==0:break
                    while True:
                        try:pending.append(front.incoming.get_nowait())
                        except queue.Empty:break
                    candidates=[tuple(payload[5+4*i:9+4*i]) for i in range(b)]
                    flags=[]
                    for r,p,l,s in candidates:
                        if r<0:flags.append(0);continue
                        item=next((w for w in pending if (w.request,w.site,w.loop,w.step)==(r,sites[p],l,s)),None)
                        flags.append(int(item is not None and item.ready.query()))
                    masks=torch.tensor(flags,device='cuda',dtype=torch.int32)
                    dist.all_reduce(masks,op=dist.ReduceOp.MIN)
                    globally_ready={item for item,ok in zip(candidates,masks.tolist()) if ok and item[0]>=0}
                    if ep.rank==0:
                        row=observations[-1]
                        row['rank0_ready']=row['ready']
                        row['ready']=[w for w in row['ready'] if (w['request'],sites.index(w['site']),w['loop'],w['step']) in globally_ready]
                        row['chosen_host_s']=time.perf_counter()-origin
                        for item in row['ready']:item['all_rank_ready_observed_host_s']=row['chosen_host_s']
                        row['readiness_scope']='All-rank device-ready quorum at one control epoch. Conservative sampled readiness, not an exhaustive continuous-time ceiling.'
                    req,site_index,loop,step=command[1:]
                    site=sites[site_index]
                    match=[w for w in pending if (w.request,w.site,w.loop,w.step)==(req,site,loop,step)]
                    while not match:
                        pending.append(front.incoming.get(timeout=180))
                        match=[w for w in pending if (w.request,w.site,w.loop,w.step)==(req,site,loop,step)]
                    work=match[0]
                    pending.remove(work)
                    stream=torch.cuda.current_stream()
                    stream.wait_event(work.ready)
                    for value in [work.hidden,work.ids,work.weights]:value.record_stream(stream)
                    if ep.rank==0:
                        ids=work.ids.reshape(-1,2)
                        scores=work.weights.reshape(-1,2)
                        sample=torch.linspace(0,len(ids)-1,min(4,len(ids)),device='cuda').long()
                        source=torch.arange(len(ids),device='cuda')[:,None]%ep.world
                        remote=int((source!=ids%ep.world).sum())
                        observations[-1]['selected_work']=dict(request=req,site=site,loop=loop,step=step,
                            shape=list(work.hidden.shape),dtype=str(work.hidden.dtype),sampled_positions=sample.cpu().tolist(),
                            sampled_ids=ids[sample].cpu().tolist(),sampled_weights=scores[sample].cpu().tolist(),
                            logical_dispatch_remote_bytes=remote*work.hidden.shape[-1]*2,
                            logical_combine_remote_bytes=remote*work.hidden.shape[-1]*2,
                            logical_restoration_bytes=work.hidden.numel()*work.hidden.element_size())
                    ep.calls[site]=loop
                    begin=torch.cuda.Event(enable_timing=True)
                    end=torch.cuda.Event(enable_timing=True)
                    begin.record()
                    work.result=ep.run(site,work.hidden,work.ids,work.weights)
                    end.record()
                    if ep.rank==0:observations[-1]['service_events']=(begin,end)
                    work.complete=torch.cuda.Event()
                    work.complete.record()
                    work.signal.set()
                for thread in threads:
                    thread.join(timeout=180)
                    assert not thread.is_alive()
                assert not errors,errors
                token_exact=all(outputs[r]==expected[r] for r in range(b))
                cases.append(dict(batch=b,arrival=pattern,cap=cap,token_exact=token_exact,
                    expected=expected,actual=outputs,coalescing_enabled=False,
                    diagnostic_wall_s=time.perf_counter()-origin,
                    scope='P1 readiness observer only. Extra per-call control broadcast. Not a clean serving benchmark.'))
                assert token_exact,'Independent request execution changes output'
                if ep.rank==0:
                    for row in observations:
                        begin,end=row.pop('service_events')
                        row['ep_service_span_ms']=begin.elapsed_time(end)
                        append(target/'ready_queue.jsonl',row)
            finally:
                for block in a.sites.values():block._ep=ep
    write(target/f'ready_probe.rank{ep.rank}.json',dict(cases=cases,status='PASS'))
