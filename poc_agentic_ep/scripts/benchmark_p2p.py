"""Effective torch peer-copy service; not pure-wire bandwidth or migration."""
import json
import os
import statistics
import torch
from poc_agentic_ep.contract import MASK,write


def benchmark(output):
    assert os.environ.get('CUDA_VISIBLE_DEVICES')==MASK
    rows=[]
    for size in (9437184,64*1024*1024):
        buffers=[torch.empty(size,dtype=torch.uint8,device=f'cuda:{r}') for r in range(4)]
        for b in buffers:b.fill_(1)
        for r in range(4):torch.cuda.synchronize(r)
        for source in range(4):
            for target in range(4):
                if source==target:continue
                assert torch.cuda.can_device_access_peer(source,target)
                times=[]
                with torch.cuda.device(target):
                    for repeat in range(13):
                        begin=torch.cuda.Event(enable_timing=True);end=torch.cuda.Event(enable_timing=True)
                        begin.record();buffers[target].copy_(buffers[source],non_blocking=True);end.record();end.synchronize()
                        if repeat>=3:times.append(begin.elapsed_time(end))
                rows.append(dict(source=source,destination=target,bytes=size,ms=times,
                    median_ms=statistics.median(times),effective_gb_s=size/statistics.median(times)/1e6))
        del buffers
        for r in range(4):
            with torch.cuda.device(r):torch.cuda.empty_cache()
    write(output,dict(status='MEASURED_P2P_COPY_ONLY',physical_mask=MASK,rows=rows,
        caveat='CUDA-event torch.copy_ completion, no model active; includes runtime peer-copy service. 64MiB exceeds H100 L2;9MiB can be cache-resident. Serial pair benchmark does not validate concurrent four-rank migration.'))
    print('P2P_COMPLETE',flush=True)


if __name__=='__main__':
    import sys
    benchmark(sys.argv[1])
