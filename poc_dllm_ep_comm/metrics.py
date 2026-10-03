"""Pure offline accounting of routes captured from an actual DeepEP execution."""
import numpy as np

MASK_ID=156895
EOS_ID=156892

def traffic(ids,source_rank,row_bytes=4096,experts_per_rank=64,ranks=4):
    ids=np.asarray(ids,dtype=np.int64)
    assert ids.ndim==2 and ids.shape[1]==8 and np.all((ids>=0)&(ids<experts_per_rank*ranks))
    owners=ids//experts_per_rank
    destinations=np.stack([(owners==r).any(axis=1) for r in range(ranks)],axis=1)
    remote=destinations.copy();remote[:,source_rank]=False
    # One BF16 hidden/output row per source-token/destination, NOT per expert.
    dispatch=remote.sum(axis=1)*row_bytes
    return dict(dispatch=dispatch,combine=dispatch.copy(),total=2*dispatch,
        destinations=destinations,remote_destinations=remote,owners=owners,
        remote_paths=(owners!=source_rank).sum(axis=1),local_paths=(owners==source_rank).sum(axis=1))

def bootstrap_mean(values,seed=3407,repeats=2000):
    values=np.asarray(values,dtype=float);values=values[np.isfinite(values)]
    if not len(values):return dict(n=0,mean=None,ci95=[None,None])
    rng=np.random.default_rng(seed)
    samples=values[rng.integers(0,len(values),(repeats,len(values)))].mean(axis=1)
    return dict(n=len(values),mean=float(values.mean()),median=float(np.median(values)),
        q25=float(np.quantile(values,.25)),q75=float(np.quantile(values,.75)),
        ci95=np.quantile(samples,[.025,.975]).tolist())

def graph_stats(edges,assignments):
    edges=np.asarray(edges);loads=np.asarray(assignments,dtype=float)
    mean=loads.mean()
    return dict(edge_density=float((edges>0).sum()/edges.size),
        remote_edge_density=float(((edges>0)&~np.eye(len(edges),dtype=bool)).sum()/(len(edges)*(len(edges)-1))),
        active_destination_ranks=int((edges.sum(axis=0)>0).sum()),hottest_rank=int(loads.argmax()),
        max_rank_load=float(loads.max()),rank_max_mean=float(loads.max()/mean) if mean else 0,
        rank_cv=float(loads.std()/mean) if mean else 0)

def decision(residual,timing_share,timing_valid,physical_valid=True):
    if not physical_valid:return 'RUNTIME_BLOCKED'
    if residual<1.5:return 'NO-GO'
    if timing_valid and timing_share<.10:return 'NO-GO'
    if residual<2 or not timing_valid or timing_share<.15:return 'CONDITIONAL'
    return 'GO'
