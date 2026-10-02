"""Exact greedy accounting and physical route features; no latency simulator."""
import numpy as np
from poc_dispatch_combine_asymmetry.accounting import account


def accept_prefix(draft, emitted):
    """Uncapped native sampler output = accepted prefix + correction/bonus."""
    emitted=[int(x) for x in emitted if x>=0]
    if not emitted:raise ValueError('no target progress')
    accepted=len(emitted)-1
    if accepted>len(draft) or emitted[:accepted]!=list(draft[:accepted]):
        raise ValueError('invalid longest-prefix sampler accounting')
    if accepted<len(draft) and emitted[-1]==draft[accepted]:
        raise ValueError('correction agrees with allegedly rejected candidate')
    return dict(accepted=accepted,sampled_progress=len(emitted),
                correction=accepted<len(draft),bonus=accepted==len(draft),
                rejected_candidates=len(draft)-accepted)


def route_features(rank_ids, valid_counts, accepted):
    """Each rank's rows are source-SP partitions, not replicated TP routes.

    Logical bytes are destination-deduplicated BF16 hidden rows. Includes
    local useful rows in all-byte totals; remote-byte totals exclude local.
    Metadata, padding headers and actual transport/HBM transactions are not
    inferred. Dummy/padding executed paths are reported separately.
    """
    traffic=np.zeros((4,4),dtype=np.int64)
    actual_traffic=np.zeros((4,4),dtype=np.int64)
    rejected=np.zeros((4,4),dtype=np.int64)
    hist=np.zeros(128,dtype=np.int64)
    actual_hist=np.zeros(128,dtype=np.int64)
    remote_pairs=local_pairs=rows=0
    fanout=[]
    for source,raw in enumerate(rank_ids):
        raw=np.asarray(raw,dtype=np.int64)
        actual_hist+=np.bincount(raw[raw>=0],minlength=128)
        actual_traffic[source]=account(raw,source)['dispatch_bytes_per_destination']
        ids=raw[:valid_counts[source]]
        if ids.size and ((ids<0).any() or (ids>=128).any()):
            raise ValueError('invalid user expert route; dummy rows must be accounted separately')
        a=account(ids,source)
        hist+=np.asarray(a['expert_histogram'])
        traffic[source]=a['dispatch_bytes_per_destination']
        rows+=len(ids);fanout+=a['distinct_destinations_per_token']
        destinations=ids//32
        local_pairs+=int((destinations==source).sum())
        remote_pairs+=int((destinations!=source).sum())
        # DP0 TP row offset: root at index0; rejected candidate input starts
        # at accepted+1, not at accepted (which predicts the correction).
        offset=source*len(raw) if source<2 else 0
        mask=np.arange(len(ids))+offset>accepted
        rejected[source]=account(ids[mask],source)['dispatch_bytes_per_destination']
    per_rank=hist.reshape(4,32).sum(axis=1)
    active=(hist.reshape(4,32)>0).sum(axis=1)
    sent,recv=traffic.sum(axis=1),traffic.sum(axis=0)
    maxb=int(max(sent.max(),recv.max()))
    meanb=float((sent.sum()+recv.sum())/8)
    remote_bytes=int(traffic.sum()-np.trace(traffic))
    return dict(token_rows=rows,valid_paths=int(hist.sum()),executed_paths=int(actual_hist.sum()),
        dummy_padding_paths=int(actual_hist.sum()-hist.sum()),unique_experts=int((hist>0).sum()),
        expert_histogram=hist.tolist(),actual_expert_histogram=actual_hist.tolist(),
        actual_unique_experts=int((actual_hist>0).sum()),
        actual_active_experts_per_rank=(actual_hist.reshape(4,32)>0).sum(axis=1).tolist(),
        actual_traffic_matrix=actual_traffic.tolist(),
        actual_max_rank_bytes=int(max(actual_traffic.sum(axis=0).max(),actual_traffic.sum(axis=1).max())),
        actual_remote_dispatch_bytes=int(actual_traffic.sum()-np.trace(actual_traffic)),
        active_experts_per_rank=active.tolist(),expert_assignments_per_rank=per_rank.tolist(),
        expert_skew=float(per_rank.max()/max(per_rank.mean(),1)),
        remote_routed_paths=remote_pairs,local_routed_paths=local_pairs,
        remote_destination_dedup_rows=remote_bytes//4096,
        rank_fanout=float(np.mean(fanout)) if fanout else 0.,
        source_destination_edges=int((traffic>0).sum()),
        logical_dispatch_bytes=int(traffic.sum()),logical_combine_bytes=int(traffic.sum()),
        logical_remote_dispatch_bytes=remote_bytes,logical_remote_combine_bytes=remote_bytes,
        per_rank_send_bytes=sent.tolist(),per_rank_receive_bytes=recv.tolist(),
        max_rank_bytes=maxb,mean_rank_bytes=meanb,rank_byte_skew=maxb/max(meanb,1),
        critical_destination_rank=int(np.argmax(recv)),
        rejected_logical_dispatch_bytes=int(rejected.sum()),
        traffic_matrix=traffic.tolist(),scope='destination_dedup useful BF16 hidden bytes, not wire/HBM')


def critical(frames):
    """One cumulative critical rank; never sum different component maxima."""
    if len(frames)!=4:raise ValueError('need all four physical EP ranks')
    if any(len(f['layers'])!=48 for f in frames):raise ValueError('incomplete target forward')
    sums=[sum(l['moe_ms'] for l in f['layers']) for f in frames]
    rank=int(np.argmax(sums));f=frames[rank]
    return dict(critical_rank=rank,
        router_selection_ms=sum(l.get('router_selection_ms',0.) for l in f['layers']),
        dispatch_ms=sum(l['dispatch_ms'] for l in f['layers']),
        expert_ms=sum(l['expert_ms'] for l in f['layers']),
        combine_ms=sum(l['combine_ms'] for l in f['layers']),
        moe_ms=sums[rank],target_forward_ms=f['target_forward_ms'],
        rule='ONE rank maximizing cumulative 48-layer D+E+C; all components on this rank')
