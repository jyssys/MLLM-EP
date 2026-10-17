"""Real four-rank collectives. Payload volume is not wire-byte accounting."""
import os
import statistics
import torch
import torch.distributed as dist
from poc_recurrent_asymmetry.contract import *


def main():
    assert os.environ['CUDA_VISIBLE_DEVICES']==MASK
    rank=int(os.environ['LOCAL_RANK']);torch.cuda.set_device(rank)
    dist.init_process_group('nccl');assert dist.get_world_size()==4
    assert str(torch.cuda.get_device_properties(rank).uuid).removeprefix('GPU-')==UUIDS[rank]
    rows=[]
    for count in [2048,32*2048,128*2048,512*2048,1024*2048]:
        for kind in ['all_reduce','broadcast']:
            x=torch.ones(count,device='cuda',dtype=torch.bfloat16)
            times=[]
            for repeat in range(35):
                x.fill_(rank+1);dist.barrier();torch.cuda.synchronize()
                a=torch.cuda.Event(enable_timing=True);b=torch.cuda.Event(enable_timing=True)
                a.record()
                work=dist.all_reduce(x,async_op=True) if kind=='all_reduce' else dist.broadcast(x,src=0,async_op=True)
                work.wait();b.record();torch.cuda.synchronize()
                if repeat>=5:times.append(a.elapsed_time(b))
                assert bool((x==(10 if kind=='all_reduce' else 1)).all())
            gathered=[None]*4;dist.all_gather_object(gathered,times)
            if rank==0:
                critical=max(range(4),key=lambda i:sum(gathered[i]))
                rows.append(dict(collective=kind,tensor_payload_bytes=count*2,
                    rank_local_ms=gathered,cumulative_critical_rank=critical,
                    median_ms=statistics.median(gathered[critical]),repeats=30,warmup=5,
                    boundary='Local CUDA events on one cumulative-critical rank, after ready barrier through completed collective.',
                    wire_bytes='UNMEASURED. Tensor payload is not NCCL algorithm traffic.',
                    label='MEASURED_PHYSICAL_FOUR_GPU_NCCL_COMPONENT'))
    if rank==0:
        output=Path(os.environ['LOOPED_KV_OUTPUT']);write(output/'collectives.json',rows)
        write(output/'status.json',dict(success=True,world_size=4,physical_GPUs=GPUS))
    dist.destroy_process_group()


if __name__=='__main__':main()
