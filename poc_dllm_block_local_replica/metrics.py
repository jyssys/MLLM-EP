"""Pure layer-specific destination-closure accounting (no latency inference)."""
from collections import Counter
import numpy as np
from poc_dllm_ep_comm.metrics import traffic, bootstrap_mean
from poc_dllm_block_local_replica.contract import ROW_DC_BYTES


def path_mask(experts):
    result=0
    for e in experts: result |= 1 << int(e)
    return result


def members(mask):
    while mask:
        bit=mask & -mask
        yield bit.bit_length()-1
        mask ^= bit


def closure_groups(ids, source):
    groups=Counter()
    for row in ids:
        for dst in range(4):
            if dst==source: continue
            subset=row[row//64==dst]
            if len(subset): groups[(dst,path_mask(subset))] += ROW_DC_BYTES
    return groups


def post_traffic(ids,source,replicas):
    """All K8 paths executed freshly; only physical owner changes."""
    native=ids//64
    owner=native.copy()
    chosen=np.array([e in replicas for e in ids.flatten()],dtype=bool).reshape(ids.shape)
    owner[chosen]=source
    dest=np.stack([(owner==r).any(axis=1) for r in range(4)],axis=1)
    dest[:,source]=False
    return dict(dc_bytes=int(dest.sum())*ROW_DC_BYTES,
                load=np.bincount(owner.flatten(),minlength=4),
                remote_paths=int((owner!=source).sum()),
                localized_paths=int((chosen & (native!=source)).sum()))


def load_stats(load):
    load=np.asarray(load,dtype=float)
    return dict(max=float(load.max()),mean=float(load.mean()),
                max_mean=float(load.max()/load.mean()) if load.mean() else 0.,
                cv=float(load.std()/load.mean()) if load.mean() else 0.,
                hottest=int(load.argmax()))
