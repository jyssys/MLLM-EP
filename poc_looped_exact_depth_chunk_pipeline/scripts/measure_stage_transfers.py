"""Physical NCCL hidden activation transfers with end-to-end synchronization."""
import os
import time
import statistics
import torch
import torch.distributed as dist
from poc_looped_exact_depth_chunk_pipeline.contract import *


def main():
    assert os.environ['CUDA_VISIBLE_DEVICES']=='4,5,6,7'
    rank=int(os.environ['LOCAL_RANK']);torch.cuda.set_device(rank)
    dist.init_process_group('nccl');assert dist.get_world_size()==4
    assert str(torch.cuda.get_device_properties(rank).uuid).removeprefix('GPU-')==UUIDS[rank]
    results=[]
    for B in [256,512,1024,2048,4096]:
        send=torch.ones((B,2048),device='cuda',dtype=torch.bfloat16)*(rank+1);recv=torch.empty_like(send)
        for name,edges in [('PP4_edge01',[(0,1)]),('PP4_edge12',[(1,2)]),('PP4_edge23',[(2,3)]),
                           ('PP2_TP2_duplicate_boundary',[(0,2),(1,3)])]:
            records=[]
            for repeat in range(33):
                dist.barrier();torch.cuda.synchronize();start=time.perf_counter_ns()
                pending=[]
                for source,destination in edges:
                    if rank==source:pending.append(dist.isend(send,destination))
                    if rank==destination:pending.append(dist.irecv(recv,source))
                for work in pending:work.wait()
                torch.cuda.synchronize();dist.barrier();wall=(time.perf_counter_ns()-start)/1e6
                if repeat>=3:records.append(wall)
                for source,destination in edges:
                    if rank==destination:assert bool((recv==source+1).all())
            gathered=[None]*4;dist.all_gather_object(gathered,records)
            if rank==0:
                # Each time comes from one local host clock and complete barrier path.
                critical=[max(values) for values in zip(*gathered)]
                results.append(dict(rows=B,mode=name,edges=edges,payload_bytes=B*2048*2*len(edges),
                    per_repeat_ms=critical,mean_ms=statistics.mean(critical),median_ms=statistics.median(critical),
                    warmup=3,repeats=30,label='MEASURED_COMPONENT',
                    boundary='Ready after initial barrier through send/recv, device synchronization, and final barrier.',
                    limitation='Conservative world-barrier coordination, not an integrated pipeline event fast path. No remote KV transfer.'))
    if rank==0:
        output=Path(os.environ['LOOPED_KV_OUTPUT']);write(output/'transfers.json',results)
        write(output/'status.json',dict(success=True))
    dist.destroy_process_group()


if __name__=='__main__':main()
