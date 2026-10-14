"""Strong multistart global load-proxy static, NOT a certified latency optimum.

The objective is sum of invocation-wise rank maxima (not aggregate balance).
Deduplicated remote destination rows break primary-objective ties. Only the
eight calibration episodes may be used to freeze this mapping.
"""
import gzip
import json
import random
import numpy as np
from poc_looped_moe_ep.contract import *


def score(counts,placement,pairs=None):
    loads=np.stack([counts[:,np.asarray(placement)==r].sum(1) for r in range(4)],axis=1)
    first=int(loads.max(1).sum())
    if pairs is None:return first
    source,e1,e2,frequency=pairs
    owners=np.asarray(placement);a=owners[e1];b=owners[e2]
    remote=((a!=source).astype(int)+((b!=source)&(b!=a)).astype(int))
    return first,int((remote*frequency).sum())


def optimize(counts,pairs,seed=3407,restarts=12,iterations=8000):
    rng=random.Random(seed);native=[e//16 for e in range(64)]
    popularity=counts.sum(0);global_static=[-1]*64;load=[0]*4;capacity=[0]*4
    for e in np.argsort(-popularity,kind='stable'):
        owner=min([r for r in range(4) if capacity[r]<16],key=lambda r:(load[r],r))
        global_static[int(e)]=owner;load[owner]+=int(popularity[e]);capacity[owner]+=1
    starts=[native,global_static]
    for _ in range(restarts):
        candidate=native.copy();rng.shuffle(candidate);starts.append(candidate)
    best=None;ledger=[]
    for start in starts:
        placement=np.asarray(start);current=score(counts,placement,pairs)
        loads=np.stack([counts[:,placement==r].sum(1) for r in range(4)],axis=1)
        for _ in range(iterations):
            a,b=rng.sample(range(64),2);ra,rb=int(placement[a]),int(placement[b])
            if ra==rb:continue
            delta=counts[:,b]-counts[:,a]
            proposed=loads.copy();proposed[:,ra]+=delta;proposed[:,rb]-=delta
            first=int(proposed.max(1).sum())
            if first>current[0]:continue
            placement[a],placement[b]=rb,ra
            trial=score(counts,placement,pairs)
            if trial<=current:current=trial;loads=proposed
            else:placement[a],placement[b]=ra,rb
        ledger.append(dict(objective=list(current),placement=placement.tolist()))
        if best is None or current<best[0]:best=(current,placement.tolist())
    return dict(placement=best[1],objective=list(best[0]),multistart=ledger,
                native_objective=list(score(counts,native,pairs)),
                global_frequency_objective=list(score(counts,global_static,pairs)),
                global_frequency_placement=global_static)


def main():
    target=ROOT/'results/frozen_best_static.json'
    assert not target.exists(),'Frozen before held-out; never retune/overwrite'
    manifest=json.loads((ROOT/'results/frozen_requests.json').read_text())
    cal=[r['request_id'] for r in manifest['requests'] if r['split']=='calibration']
    counts=[];pair_hist={}
    directory=ROOT/'results/raw/calibration_S0'
    for request in cal:
        rank_records=[]
        for rank in range(4):
            path=directory/f'{request}_trace_6.rank{rank}.events.json.gz'
            rank_records.append(json.load(gzip.open(path,'rt')))
        for forward in range(3):
            for loop in range(16):
                histogram=np.zeros(64,dtype=np.int64)
                for rank,records in enumerate(rank_records):
                    row=next(r for r in records if r['forward']==forward and r['loop']==loop and r['pool']=='core')
                    ids=np.asarray(row['ids']);histogram+=np.bincount(ids.ravel(),minlength=64)
                    for e1,e2 in ids:
                        key=(rank,int(e1),int(e2));pair_hist[key]=pair_hist.get(key,0)+1
                counts.append(histogram)
    pairs=np.asarray([(*key,value) for key,value in pair_hist.items()]).T
    result=optimize(np.asarray(counts),pairs)
    result.update(calibration_request_ids=cal,calibration_requests=8,held_out_requests_used=0,
        seed=3407,native_H=16,expert_capacity_per_rank=16,
        objective_domain='STRUCTURAL assignment-path units; remote deduplicated rows only as tiebreak',
        label='S1_BEST_FOUND_GLOBAL_STATIC_PROXY_NOT_CERTIFIED_LATENCY_OPTIMUM',
        optimization='14 starts ×8000 bounded pair swaps; all-loop invocation-wise maxima',
        native_reference='S0 contiguous 16 experts/rank',frozen_before_held_out=True)
    write(target,result);print({k:result[k] for k in ['objective','native_objective','global_frequency_objective']})


if __name__=='__main__':main()
