"""One immutable request plan; history-only guards, no future pruning for causal policies."""
import collections
import numpy as np
from poc_dllm_request_persistent_replica.fast_bundle import bundle_greedy
from poc_dllm_block_local_replica.scripts.simulate_replication import choose_frequency
from poc_dllm_request_persistent_replica.data import Profile


def plan(history,slots,cap=None,groups_override=None,freq_override=None,frequency_only=False):
    groups=history.groups() if groups_override is None else groups_override
    freq=history.frequency() if freq_override is None else freq_override
    current=history.load.copy().astype(np.int64)
    maximum=history.load.max(axis=2)*(1+cap) if cap is not None else None
    chosen={s:{} for s in range(4)}
    for s in range(4):
        def allowed(layer,experts):
            # Only this source's load can increase; remote owners decrease.
            # All previous accepted proposals already satisfy the same cap.
            # Therefore this test is exactly the original four-rank maximum
            # check, without allocating a time×4 delta for rejected offers.
            source_add=history.counts[:,layer-1,s,experts].sum(axis=1,dtype=np.int64)
            if cap is not None and np.any(current[:,layer-1,s]+source_add>maximum[:,layer-1]+1e-9):return False
            current[:,layer-1,s]+=source_add
            for e in experts:current[:,layer-1,e//64]-=history.counts[:,layer-1,s,e]
            return True
        if not frequency_only:
            chosen[s]=bundle_greedy(groups[s],slots,allowed if cap is not None else None)
        else:
            items=[(freq[l-1,s,e],l,e) for l in range(1,20) for e in range(256) if e//64!=s and freq[l-1,s,e]>0]
            items.sort(key=lambda x:(-x[0],x[1],x[2]))
            count=0
            for _,l,e in items:
                if count==slots:break
                if cap is None or allowed(l,[e]):
                    chosen[s].setdefault(l,set()).add(e);count+=1
    return chosen


def global_inputs(calibration_profiles):
    """Each calibration request has equal feature weight; layer identity intact."""
    groups={s:{l:collections.Counter() for l in range(1,20)} for s in range(4)}
    frequency=np.zeros((19,4,256))
    blocks=[]
    for p in calibration_profiles:
        counts=p.frequency().astype(float)
        sums=counts.sum(axis=2,keepdims=True)
        frequency+=np.divide(counts,sums,out=np.zeros_like(counts),where=sums>0)/len(calibration_profiles)
        pg=p.groups()
        for s in range(4):
            for l in range(1,20):
                norm=sum(pg[s][l].values())
                for key,value in pg[s][l].items():
                    if norm:groups[s][l][key]+=value/norm/len(calibration_profiles)
        blocks.extend(p.blocks)
    return Profile(blocks),groups,frequency
