"""Include fixed per-cap migration fractions before attributing policy gains."""
import collections
import itertools
import math
from .replay import run,evaluate


def frontier(model,workload):
    caps=sorted({max(g.caps) for g in workload})
    occurrences=collections.Counter(max(g.caps) for g in workload)
    actions={cap:[(0,0,1)]+[(m,k,occurrences[cap]) for m in (2,4,8,12)
        for k in range(1,occurrences[cap]+1)] for cap in caps}
    best=None;best95=None;tested=0
    for selected in itertools.product(*(actions[cap] for cap in caps)):
        mapping=dict(zip(caps,selected));seen=collections.Counter();plan=[]
        for group in workload:
            cap=max(group.caps);m,k,n=mapping[cap];j=seen[cap];seen[cap]+=1
            plan.append(m if (j+1)*k//n>j*k//n else 0)
        state=run(model,workload,lambda i,_g:plan[i]);metric=evaluate(state)
        score=(metric['SLO_output_tokens_s'],-metric['TTFT_p95_ms'])
        tested+=1
        if best is None or score>best[0]:
            best=(score,dict(metric=metric,plan=plan,policy=dict(kind='PER_CAP_FIXED_FRACTION',
                mapping={str(cap):dict(layers=m,migrated_groups_per_cycle=k,cycle_groups=n)
                    for cap,(m,k,n) in mapping.items()})))
        if metric['Joint_attainment']>=.95 and (best95 is None or score>best95[0]):
            best95=(score,dict(metric=metric,plan=plan,policy=dict(kind='PER_CAP_FIXED_FRACTION',
                mapping={str(cap):dict(layers=m,migrated_groups_per_cycle=k,cycle_groups=n)
                    for cap,(m,k,n) in mapping.items()})))
    return dict(best=best[1],best_with_Joint95=best95[1] if best95 else None,tested=tested,
        scope='Known output caps, fixed occurrence counters and fixed phase0. No arrival-time or completion-state input. Complete fractions for the observed class counts. This is stronger than one cut per cap, but not all possible static policies or phases.')


def apply(policy,caps,size=8):
    seen=collections.Counter();result=[]
    for i in range(0,len(caps),size):
        cap=max(caps[i:i+size]);item=policy['mapping'][str(cap)];j=seen[cap];seen[cap]+=1
        k,n=item['migrated_groups_per_cycle'],item['cycle_groups']
        m=item['layers'] if (j+1)*k//n>j*k//n else 0
        result.extend([m]*min(size,len(caps)-i))
    return result
