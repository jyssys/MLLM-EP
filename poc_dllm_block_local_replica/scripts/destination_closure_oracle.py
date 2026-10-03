"""Future-truth destination-bundle greedy with an optional LP upper bound.

Replica x(layer,expert) and closure y(layer,destination,expert-set) variables
obey y<=x for every selected expert. A shared budget spans ALL layers/rank.
The LP relaxation certifies an upper bound; greedy is an achievable oracle
reference, not a claim of exact combinatorial optimum.
"""
from collections import Counter
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix
from poc_dllm_block_local_replica.metrics import members


def bundle_greedy(groups,slots,allowed=None):
    chosen={layer:0 for layer in groups}
    remaining={layer:dict(counter) for layer,counter in groups.items()}
    used=0
    while used<slots:
        shortlist=[]
        for layer,values in remaining.items():
            merged=Counter()
            for (_,mask),weight in values.items():
                need=mask & ~chosen[layer]
                if need:merged[need]+=weight
            for need,weight in merged.items():
                cost=need.bit_count()
                if cost<=slots-used:shortlist.append((weight/cost,layer,need))
        shortlist.sort(key=lambda x:(-x[0],x[1],x[2]))
        best=None
        # Eight leading bundles; enough to account for cross-bundle closure
        # without constructing a learned/complex planner. Fixed before eval.
        for _,layer,need in shortlist[:8]:
            new=chosen[layer]|need
            gain=sum(weight for (_,mask),weight in remaining[layer].items() if mask & ~new==0)
            density=gain/need.bit_count()
            if best is None or density>best[0]:best=(density,layer,need,gain)
        if best is None:break
        _,layer,need,_=best
        if allowed is not None and not allowed(layer,list(members(need))):
            # Do not keep retrying a load-violating bundle. Remove only its
            # planning offer; preserve true groups for final accounting.
            for key in list(remaining[layer]):
                if key[1] & ~chosen[layer]==need:del remaining[layer][key]
            continue
        chosen[layer]|=need;used+=need.bit_count()
        remaining[layer]={key:value for key,value in remaining[layer].items() if key[1]&~chosen[layer]}
    return {layer:set(members(mask)) for layer,mask in chosen.items() if mask}


def lp_upper(groups,slots):
    xs=sorted({(layer,e) for layer,values in groups.items() for (_,mask) in values for e in members(mask)})
    yi=[(layer,key,weight) for layer,values in groups.items() for key,weight in values.items()]
    if not xs:return 0.
    lookup={key:i for i,key in enumerate(xs)};n=len(xs)+len(yi)
    row=[];col=[];val=[];b=[float(slots)]
    for i in range(len(xs)):row.append(0);col.append(i);val.append(1.)
    c=np.zeros(n)
    for j,(layer,(_,mask),weight) in enumerate(yi):
        c[len(xs)+j]=-weight
        for e in members(mask):
            k=len(b);b.append(0.)
            row.extend([k,k]);col.extend([len(xs)+j,lookup[(layer,e)]]);val.extend([1.,-1.])
    mat=coo_matrix((val,(row,col)),shape=(len(b),n)).tocsr()
    result=linprog(c,A_ub=mat,b_ub=np.array(b),bounds=(0.,1.),method='highs',options={'time_limit':5.})
    return float(-result.fun) if result.success else None
