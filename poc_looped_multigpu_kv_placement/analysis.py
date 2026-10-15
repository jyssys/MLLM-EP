"""Logical cache references and request-level statistics, not latency oracles."""
import csv
import numpy as np


def kv_bytes_per_token(slots,kv_heads,head_dim,dtype_bytes=2,tp=1):
    assert kv_heads%tp==0,'No fractional native KV heads'
    return 2*slots*(kv_heads//tp)*head_dim*dtype_bytes


def bootstrap(values,seed=41015):
    values=np.asarray(values,dtype=float);rng=np.random.default_rng(seed)
    assert len(values)>0
    means=values[rng.integers(0,len(values),size=(2000,len(values)))].mean(axis=1)
    return dict(N=len(values),mean=float(values.mean()),median=float(np.median(values)),
        CI95=[float(x) for x in np.quantile(means,[.025,.975])],unit='request',resamples=2000)


def table(path,rows):
    rows=list(rows);keys=list(dict.fromkeys(key for row in rows for key in row))
    with path.open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=keys);writer.writeheader();writer.writerows(rows)
