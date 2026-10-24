"""NCCL P2P transfer and collective diagnostics over real model KV pages."""
import time
import numpy as np
import torch
import torch.distributed as dist
from .common import FIXTURES,write

def run(target,rank,world):
    packet=torch.load(FIXTURES/'P32768_d1_l12.pt',map_location='cpu',weights_only=True)
    rows=[]
    for length in (1024,2048,8192,16384):
        original=torch.stack((packet['k'][:length],packet['v'][:length])).contiguous()
        source=original.to('cuda');received=torch.empty_like(source)
        samples=[]
        for repeat in range(15):
            dist.barrier();torch.cuda.synchronize();start=time.perf_counter()
            request=dist.isend(source,dst=1) if rank==0 else dist.irecv(received,src=0)
            request.wait();dist.barrier();torch.cuda.synchronize()
            samples.append((time.perf_counter()-start)*1000)
        if rank==1:assert torch.equal(received.cpu(),original)
        rows.append(dict(tokens=length,bytes=source.numel()*source.element_size(),wall_ms=samples,
            median_ms=float(np.median(samples)),GiB_s_including_sync=source.numel()*source.element_size()/2**30/(np.median(samples)/1000),
            byte_exact=rank==1,evidence='LIVE_MEASURED_OPERATOR',
            scope='NCCL P2P plus completion barriers. Not raw NVLink link bandwidth.'))
        write(target/f'rank{rank}_peer_transfers.json',rows)
    samples=[];tensor=torch.ones(2048,device='cuda',dtype=torch.float32)
    for repeat in range(20):
        tensor.fill_(1);dist.barrier();torch.cuda.synchronize();start=time.perf_counter()
        dist.all_reduce(tensor);torch.cuda.synchronize();samples.append((time.perf_counter()-start)*1000)
    assert torch.all(tensor==world)
    write(target/f'rank{rank}_allreduce.json',dict(bytes=8192,world=world,wall_ms=samples,
        evidence='LIVE_MEASURED_OPERATOR',scope='Standalone NCCL collective. Not model TP latency.'))
