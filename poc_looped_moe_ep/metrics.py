"""Destination-deduplicated accounting generalized to Foil's native top-2."""
import numpy as np


def traffic(ids,source,placement,row_bytes=2048):
    ids=np.asarray(ids,dtype=int);owners=np.asarray(placement)[ids]
    destinations=np.stack([(owners==r).any(1) for r in range(4)],axis=1)
    remote=destinations.copy();remote[:,source]=False
    expert_counts=np.bincount(ids.ravel(),minlength=len(placement))
    return dict(owners=owners,destinations=destinations,remote=remote,
        dispatch_bytes=int(remote.sum())*row_bytes,combine_bytes=int(remote.sum())*row_bytes,
        remote_paths=int((owners!=source).sum()),expert_counts=expert_counts,
        rank_counts=np.bincount(owners.ravel(),minlength=4))


def jsd(p,q):
    p=np.asarray(p,dtype=float);q=np.asarray(q,dtype=float)
    p=p/p.sum();q=q/q.sum();m=(p+q)/2
    def kl(a):
        valid=a>0
        return float(np.sum(a[valid]*np.log2(a[valid]/m[valid])))
    return (kl(p)+kl(q))/2


def balanced_candidate(counts,fallback):
    """Future-truth load heuristic; not a physical latency oracle."""
    placement=[-1]*len(counts);loads=[0]*4;capacity=[0]*4
    for expert in np.argsort(-np.asarray(counts),kind='stable'):
        owner=min([r for r in range(4) if capacity[r]<len(counts)//4],key=lambda r:(loads[r],r))
        placement[int(expert)]=owner;loads[owner]+=int(counts[expert]);capacity[owner]+=1
    old=[sum(counts[e] for e in range(len(counts)) if fallback[e]==r) for r in range(4)]
    return placement if max(loads)<=max(old) else list(fallback)
