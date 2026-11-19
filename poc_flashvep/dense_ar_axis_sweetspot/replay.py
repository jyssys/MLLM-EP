"""Update FIFO queues and whole-cohort guards after each modeled action."""
from dataclasses import dataclass
import heapq
import math


@dataclass(frozen=True)
class Cost:
    P: float
    preparation: float
    decode: float
    P_gap: float = 0.
    D_gap: float = 0.

    def validate(self):
        assert min(self.P,self.preparation,self.decode,self.P_gap,self.D_gap)>=0
        assert all(math.isfinite(x) for x in (self.P,self.preparation,self.decode,self.P_gap,self.D_gap))


def simulate(arrivals, costs, batch=1, window=32, choose=None, kv_slots=None):
    """Times use seconds. Costs include measured host boundaries, not GPU time."""
    assert len(arrivals)==len(costs) and len(arrivals)%batch==0 and window>=batch
    assert all(b>=a for a,b in zip(arrivals,arrivals[1:]))
    for c in costs:
        c.validate()
    n=len(arrivals)
    costs=list(costs)
    if kv_slots is not None:
        assert kv_slots>=batch
    queue=[]
    for i,t in enumerate(arrivals):
        heapq.heappush(queue,(t,0,i))
    pending=[]
    ready=[]
    result=[dict(arrival=t) for t in arrivals]
    P_busy=D_busy=False
    outstanding=reserved=0
    D_end=0.
    cohort=[]
    while queue:
        t,kind,index=heapq.heappop(queue)
        if kind==0:
            pending.append(index)
        elif kind==1:
            P_busy=False
            result[index]['P_ready']=t
            outstanding+=1
            ready.append(index)
        elif kind==2:
            D_busy=False
            for i in cohort:
                result[i]['finish']=t
            outstanding-=len(cohort)
            reserved-=len(cohort)
            cohort=[]
        elif kind==3:
            P_busy=False
        # The guard reserves the whole cohort until its last client completion.
        if not D_busy and len(ready)>=batch:
            cohort=ready[:batch]
            del ready[:batch]
            D_busy=True
            start=t+max(costs[i].D_gap for i in cohort)
            preparation=sum(costs[i].preparation for i in cohort)
            decode=max(costs[i].decode for i in cohort)
            first=start+preparation
            finish=first+decode
            D_end=finish
            for i in cohort:
                result[i].update(D_start=start,first=first,cohort=tuple(cohort))
            heapq.heappush(queue,(finish,2,cohort[0]))
        if not P_busy and pending and outstanding<window and (kv_slots is None or reserved<kv_slots):
            i=pending.pop(0)
            if choose is not None:
                costs[i]=choose(i,t,dict(pending_source_indices=tuple(pending),
                    ready_indices=tuple(ready),active_indices=tuple(cohort),
                    estimated_active_D_remaining=max(0.,D_end-t) if D_busy else 0.,
                    outstanding=outstanding,reserved_KV_slots=reserved))
                costs[i].validate()
            P_busy=True
            reserved+=1
            start=t+costs[i].P_gap
            result[i].update(P_start=start,reserved_KV_slots_at_P_start=reserved,
                P_pending_at_P_start=len(pending),D_pending_at_P_start=len(ready))
            heapq.heappush(queue,(start+costs[i].P,1,i))
    assert not pending and not ready and outstanding==reserved==0
    assert all('finish' in r for r in result)
    for r in result:
        r.update(TTFT=r['first']-r['arrival'],P_wait=r['P_start']-r['arrival'],
                 Ready_D_wait=r['D_start']-r['P_ready'])
        assert min(r['TTFT'],r['P_wait'],r['Ready_D_wait'])>=-1e-10
    return result


def summary(rows, output, TTFT_limit=5., ITL_limit=.034, TPOT_limit=.033):
    from ..dense_ar_slo_layer_pd.analyze import percentile
    duration=max(r['finish'] for r in rows)-min(r['arrival'] for r in rows)
    passed=[]
    for r in rows:
        interval=(r['finish']-r['first'])/max(output-1,1)
        passed.append(r['TTFT']<=TTFT_limit and interval<=ITL_limit and interval<=TPOT_limit)
    return dict(TTFT_mean_ms=1000*sum(r['TTFT'] for r in rows)/len(rows),
        TTFT_p95_ms=1000*percentile([r['TTFT'] for r in rows],.95),
        Ready_D_wait_mean_ms=1000*sum(r['Ready_D_wait'] for r in rows)/len(rows),
        output_tokens_s=len(rows)*output/duration,
        qualified_output_tokens_s=sum(passed)*output/duration,
        joint_attainment=sum(passed)/len(passed),observation_seconds=duration,
        evidence='OFFLINE_REPLAY',
        limitation='Mean-cost host-boundary model. Constant Decode intervals cannot predict measured ITL tails or prove Decode noninferiority. Finite trace, not sustainable capacity.')
