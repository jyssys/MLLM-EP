import numpy as np
from poc_dllm_request_persistent_replica.data import Profile
from poc_dllm_request_persistent_replica.planner import plan
from poc_dllm_block_local_replica.scripts.destination_closure_oracle import bundle_greedy


def reference_plan(history,slots,cap):
    current=history.load.copy().astype(np.int64)
    maximum=history.load.max(axis=2)*(1+cap)
    groups=history.groups();chosen={}
    for s in range(4):
        def allowed(layer,experts):
            delta=np.zeros((history.n,4),dtype=np.int64)
            for e in experts:
                counts=history.counts[:,layer-1,s,e]
                delta[:,s]+=counts;delta[:,e//64]-=counts
            new=current[:,layer-1]+delta
            if np.any(new.max(axis=1)>maximum[:,layer-1]+1e-9):return False
            current[:,layer-1]=new;return True
        chosen[s]=bundle_greedy(groups[s],slots,allowed)
    return chosen


def test_source_only_predicate_exact_four_rank_parity():
    rng=np.random.default_rng(101)
    for _ in range(6):
        data={}
        for t in range(5):
            for l in (1,2,3):
                for source in range(4):
                    data[t,l,source]=np.array([rng.choice(256,8,replace=False) for _ in range(3)],dtype=np.int16)
        p=Profile([dict(iterations=5,data=data)])
        for cap in (.1,.2,.5):
            assert plan(p,8,cap)==reference_plan(p,8,cap)
