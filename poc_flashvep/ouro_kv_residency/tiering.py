"""Measure exact initialized BF16 page transfers. Do not build an offload cache."""
import time
import numpy as np
import torch
from .common import FIXTURES,write

def run(job,target):
    fixture=torch.load(FIXTURES/job['fixture'],map_location='cpu',weights_only=True)
    rows=[]
    for tokens in (1024,2048,8192,16384):
        source=torch.stack((fixture['k'][:tokens],fixture['v'][:tokens])).contiguous().pin_memory()
        output=torch.empty_like(source,pin_memory=True)
        gpu=torch.empty(source.shape,device='cuda',dtype=torch.bfloat16)
        measurements={}
        for mode in ('H2D','D2H','ROUNDTRIP'):
            samples=[];events=[]
            for _ in range(12):
                begin=torch.cuda.Event(enable_timing=True);end=torch.cuda.Event(enable_timing=True)
                start=time.perf_counter();begin.record()
                if mode in ('H2D','ROUNDTRIP'):gpu.copy_(source,non_blocking=True)
                if mode in ('D2H','ROUNDTRIP'):output.copy_(gpu,non_blocking=True)
                end.record();end.synchronize()
                samples.append((time.perf_counter()-start)*1000);events.append(begin.elapsed_time(end))
            measurements[mode]=dict(wall_ms=samples,cuda_ms=events,median_wall_ms=float(np.median(samples)))
        assert torch.equal(source,output)
        rows.append(dict(tokens=tokens,bytes_one_direction=source.numel()*source.element_size(),
            byte_exact=True,measurements=measurements,evidence='LIVE_MEASURED_OPERATOR',
            scope='One physical layer and one depth. Whole-depth migration requires 24 such payloads.'))
        write(target/'tiering.json',rows)
    return dict(rows=rows)
