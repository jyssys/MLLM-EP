"""Transparent admission-pinned, slot-homogeneous CDB event model.

This is trace simulation. It is not live inference or a GPU measurement.
The validation gate controls whether its oracle estimates are usable.
"""
import collections
import math
import numpy as np

def stage_features(stage,items):
    n=len(items)
    if stage=='prefill':
        q=sum(i['query'] for i in items)
        return [1,q/1024,sum(i['query']*(i['query']+i.get('past',0)) for i in items)/1048576,n]
    if stage=='recurrent':
        copies=[4-(i['depth']+1) for i in items if i.get('exited',False)]
        return [1,n,sum(i['context'] for i in items)/1024,
                int(any(c>0 for c in copies)),sum(copies)]
    return [1,n]

class CostModel:
    def __init__(self,parameters):self.parameters=parameters
    def cost(self,stage,items):
        p=self.parameters[stage]
        return max(p['minimum_ms'],sum(a*b for a,b in zip(stage_features(stage,items),p['coefficients'])))

def simulate_rank(requests,model,*,cap=16,num_blocks=2048,max_tokens=4096):
    """Preserve request ownership, autoregression, KV slot grouping, and CDB priority."""
    if not requests:return dict(outputs=[],makespan_ms=0,stages=[])
    arrivals=sorted(requests,key=lambda r:(r['arrival_ms'],r['order']))
    waiting=collections.deque()
    ready=collections.deque()
    coda=collections.deque()
    active={}
    indices={r['id']:0 for r in arrivals}
    first={}
    times={r['id']:[] for r in arrivals}
    finished={}
    started={}
    stages=[]
    t=0.
    a=0
    byid={r['id']:r for r in arrivals}
    def reserved():
        return sum(math.ceil((r['prompt_tokens']+r.get('cap',256))/16) for r in active.values())
    while a<len(arrivals) or waiting or ready or coda or active:
        while a<len(arrivals) and arrivals[a]['arrival_ms']<=t:
            waiting.append(dict(arrivals[a],remaining=arrivals[a]['prompt_tokens'],past=0))
            a+=1
        if not active and not waiting and a<len(arrivals):
            t=max(t,arrivals[a]['arrival_ms']);continue
        if coda:
            batch=[coda.popleft() for _ in range(min(cap,len(coda)))]
            duration=model.cost('coda',[dict(id=rid) for rid in batch])
            stages.append(dict(stage='coda',start_ms=t,duration_ms=duration,batch=len(batch)))
            t+=duration
            new=[]
            for rid in batch:
                times[rid].append(t)
                indices[rid]+=1
                if indices[rid]>=len(byid[rid]['depths']):
                    finished[rid]=t
                    active.pop(rid)
                else:new.append((rid,0))
            ready.extendleft(reversed(new))
            continue
        # Reserve admission and the upstream default min_free_slots=max(1, cap//8).
        free=cap-len(active)
        prefill=[]
        if waiting and free>=max(1,cap//8):
            budget=max_tokens
            for _ in range(min(free,len(waiting))):
                r=waiting[0]
                need=math.ceil((r['prompt_tokens']+r.get('cap',256))/16)
                if reserved()+need>num_blocks:break
                r=waiting.popleft()
                q=min(r['remaining'],budget)
                if q<=0:
                    waiting.appendleft(r);break
                r['remaining']-=q
                prefill.append(dict(id=r['id'],query=q,past=r['past'],final_chunk=r['remaining']==0))
                r['past']+=q
                budget-=q
                if r['remaining']:
                    # A prompt chunk retains its rank and blocks. This branch is explicitly limited
                    # by validation because the compact model does not model in-flight prefill overlap.
                    waiting.appendleft(r)
                    break
                active[r['id']]=byid[r['id']]
                started[r['id']]=t
        if prefill:
            duration=model.cost('prefill',prefill)
            stages.append(dict(stage='prefill',start_ms=t,duration_ms=duration,batch=len(prefill)))
            t+=duration
            for item in prefill:
                if not item['final_chunk']:continue
                rid=item['id'];first[rid]=t;times[rid].append(t)
                if not byid[rid]['depths']:
                    finished[rid]=t;active.pop(rid)
                else:ready.append((rid,0))
            continue
        if ready:
            rid,depth=ready.popleft()
            batch=[rid]
            deferred=collections.deque()
            while ready and len(batch)<cap:
                other,d=ready.popleft()
                if d==depth:batch.append(other)
                else:deferred.append((other,d))
            deferred.extend(ready);ready=deferred
            items=[]
            survivors=[]
            for rid in batch:
                r=byid[rid]
                target=r['depths'][indices[rid]]
                assert 2<=target<=4
                exited=depth+1>=target
                items.append(dict(id=rid,depth=depth,context=r['prompt_tokens']+indices[rid]+1,exited=exited))
                if exited:coda.append(rid)
                else:survivors.append((rid,depth+1))
            duration=model.cost('recurrent',items)
            stages.append(dict(stage='recurrent',start_ms=t,duration_ms=duration,batch=len(items),depth=depth))
            t+=duration
            ready.extendleft(reversed(survivors))
            continue
        if a<len(arrivals):t=max(t,arrivals[a]['arrival_ms']);continue
        raise RuntimeError('Unrealizable simulated transition: active requests have no legal work.')
    outputs=[dict(id=r['id'],arrival_ms=r['arrival_ms'],first_token_ms=first[r['id']],
        finish_ms=finished[r['id']],token_times_ms=times[r['id']],start_ms=started[r['id']]) for r in requests]
    return dict(outputs=outputs,makespan_ms=max(finished.values()),stages=stages)

def simulate_dp(assigned,model):
    ranks=[simulate_rank(values,model) for values in assigned]
    outputs=[v for rank in ranks for v in rank['outputs']]
    e2e=[v['finish_ms']-v['arrival_ms'] for v in outputs]
    ttft=[v['first_token_ms']-v['arrival_ms'] for v in outputs]
    itl=[b-a for r in outputs for a,b in zip(r['token_times_ms'],r['token_times_ms'][1:])]
    finish=max(r['makespan_ms'] for r in ranks)
    return dict(ranks=ranks,outputs=outputs,makespan_ms=finish,
        E2E_P50_ms=float(np.quantile(e2e,.5)),E2E_P95_ms=float(np.quantile(e2e,.95)),
        TTFT_P95_ms=float(np.quantile(ttft,.95)),ITL_P95_ms=float(np.quantile(itl,.95)) if itl else None,
        requests_per_s=len(outputs)*1000/finish)
