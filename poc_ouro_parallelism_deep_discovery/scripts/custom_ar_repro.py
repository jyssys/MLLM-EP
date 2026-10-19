"""Minimal actual graph-buffer registration regression, without a model load."""
import argparse
import os
from poc_ouro_parallelism_deep_discovery.runtime import configure_native_imports
from poc_ouro_parallelism_deep_discovery.contract import *

def main():
    p=argparse.ArgumentParser();p.add_argument('--allocator',choices=['expandable','native'],required=True)
    a=p.parse_args()
    os.environ['PYTORCH_CUDA_ALLOC_CONF']='expandable_segments:'+('True' if a.allocator=='expandable' else 'False')
    configure_native_imports()
    import torch
    import torch.distributed as dist
    from vllm.distributed.device_communicators.custom_all_reduce import CustomAllreduce
    rank=int(os.environ['LOCAL_RANK']);mask=validate_mask();torch.cuda.set_device(rank)
    dist.init_process_group('gloo')
    out=Path(os.environ['LOOPED_KV_OUTPUT'])
    write(out/f'flags_rank{rank}.json',dict(allocator=a.allocator,mask=mask,rank=rank,
        torch=torch.__version__,uuid=str(torch.cuda.get_device_properties(rank).uuid)))
    ar=CustomAllreduce(dist.group.WORLD,torch.device('cuda',rank))
    assert not ar.disabled,'A disabled custom All-Reduce is not a passing reproduction.'
    x=torch.full((1,2048),rank+1,device='cuda',dtype=torch.bfloat16)
    for _ in range(3):ar.custom_all_reduce(x)
    torch.cuda.synchronize();dist.barrier()
    graph=torch.cuda.CUDAGraph()
    with ar.capture():
        with torch.cuda.graph(graph):
            graph_input=x+0
            result=ar.custom_all_reduce(graph_input)
    graph.replay();torch.cuda.synchronize()
    assert result is not None and bool((result==sum(range(1,len(mask)+1))).all())
    write(out/f'custom_ar_rank{rank}.json',dict(success=True,allocator=a.allocator,graph_registration=True,
        expected_sum=sum(range(1,len(mask)+1)),mask=mask,rank=rank))
    dist.barrier();ar.close();dist.destroy_process_group()

if __name__=='__main__':main()
