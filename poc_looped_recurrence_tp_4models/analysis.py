"""Request-level summaries. GPU service never becomes an E2E fraction."""
import csv
import json
from pathlib import Path
import numpy as np


def read(path):
    return json.loads(Path(path).read_text())


def table(path, rows):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    if not rows:return
    fields=list(dict.fromkeys(key for row in rows for key in row))
    with path.open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=fields);writer.writeheader()
        for row in rows:
            writer.writerow({key:json.dumps(value,sort_keys=True) if isinstance(value,(dict,list)) else value
                for key,value in row.items()})


def bootstrap(values):
    values=np.asarray(values,dtype=float)
    if not len(values):return dict(mean=None,low=None,high=None,n=0)
    rng=np.random.default_rng(41015)
    samples=values[rng.integers(0,len(values),(5000,len(values)))].mean(axis=1)
    return dict(mean=float(values.mean()),low=float(np.quantile(samples,.025)),high=float(np.quantile(samples,.975)),
        n=len(values),resamples=5000,unit='Independent request after averaging technical repeats')


def category(name):
    name=name.lower()
    if 'nccl' in name or 'cross_device_reduce' in name:return 'TP_COLLECTIVE'
    if 'reshape_and_cache' in name:return 'KV_ROW_WRITE'
    if 'flashattn' in name:return 'ATTENTION_WITH_KV_READ'
    if any(word in name for word in ['nvjet','gemm','cublaslt::splitkreduce']):return 'QKV_FFN_OUTPUT_GEMM'
    if any(word in name for word in ['rms_norm','silu']):return 'NORM_ACTIVATION'
    if 'memcpy' in name or 'memset' in name:return 'MEMORY_OPERATION'
    return 'OTHER_KERNEL'


def union_ms(events):
    result=0.;end=None
    for start,stop in sorted((e['start_us'],e['end_us']) for e in events):
        if end is None or start>=end:result+=stop-start;end=stop
        elif stop>end:result+=stop-end;end=stop
    return result/1000
