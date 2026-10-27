"""Separate observer run. Kernel unions are not clean E2E percentages."""
from collections import defaultdict
import gzip
import json
from .common import write

def union(intervals):
    end=None;total=0.
    for left,right in sorted(intervals):
        if end is None or left>end:total+=right-left;end=right
        elif right>end:total+=right-end;end=right
    return total

def profile(a,ep,target):
    import torch
    import torch.distributed as dist
    from .workloads import inputs
    ids,requests=inputs(a,1,128)
    for _ in range(3):out=a.forward(ids);del out
    dist.barrier();torch.cuda.synchronize()
    ep.profiler_labels=True
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,torch.profiler.ProfilerActivity.CUDA],
            record_shapes=False,profile_memory=False,with_stack=False) as observer:
        out=a.forward(ids)
        torch.cuda.synchronize()
    ep.profiler_labels=False
    del out
    path=target/f'kineto_b1_p128.rank{ep.rank}.json'
    observer.export_chrome_trace(str(path))
    data=json.loads(path.read_text())
    kernels=[e for e in data['traceEvents'] if e.get('cat')=='kernel' and e.get('ph')=='X']
    assert kernels,'CUPTI did not capture GPU kernels. Do not report idle from empty data.'
    intervals=[(e['ts'],e['ts']+e['dur']) for e in kernels]
    compute=[(e['ts'],e['ts']+e['dur']) for e in kernels if 'nccl' not in e['name'].lower()]
    nccl=[(e['ts'],e['ts']+e['dur']) for e in kernels if 'nccl' in e['name'].lower()]
    start=min(x[0] for x in intervals);end=max(x[1] for x in intervals)
    totals=defaultdict(lambda:[0,0.])
    for e in kernels:totals[e['name']][0]+=1;totals[e['name']][1]+=e['dur']
    write(target/f'kernel_summary.rank{ep.rank}.json',dict(rank=ep.rank,batch=1,prefix=128,
        window_us=end-start,kernel_union_us=union(intervals),non_nccl_union_us=union(compute),
        nccl_union_us=union(nccl),gpu_no_kernel_us=end-start-union(intervals),
        caveat='Observer window only. NCCL kernels can include rank waiting. Concurrent intervals are unioned, not summed.',
        kernel_names=[dict(name=k,count=v[0],service_us=v[1]) for k,v in sorted(totals.items(),key=lambda kv:-kv[1][1])]))
    with path.open('rb') as source,gzip.open(str(path)+'.gz','wb') as output:
        while data_bytes:=source.read(8<<20):output.write(data_bytes)
    # This uncompressed file was created only by this function. Keep its compressed equivalent.
    path.unlink()

