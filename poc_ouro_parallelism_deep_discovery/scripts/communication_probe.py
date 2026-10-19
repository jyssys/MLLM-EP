"""Physical collective and lossless-transform costs. No serving speedup."""
import datetime
import statistics
from poc_ouro_parallelism_deep_discovery.runtime import configure_imports
from poc_ouro_parallelism_deep_discovery.contract import *
configure_imports()
import torch
import torch.distributed as dist

def timed(fn):
    for _ in range(7):fn()
    torch.cuda.synchronize();dist.barrier()
    graph=torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        for _ in range(16):answer=fn()
    graph.replay();torch.cuda.synchronize()
    values=[]
    for _ in range(31):
        dist.barrier();torch.cuda.synchronize()
        a,b=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
        a.record();graph.replay();b.record();b.synchronize()
        values.append(a.elapsed_time(b)/16)
    gathered=[None]*dist.get_world_size();dist.all_gather_object(gathered,values)
    critical=[max(r[i] for r in gathered) for i in range(len(values))]
    return dict(critical_stream_ms=critical,median_ms=statistics.median(critical),per_rank=gathered,
        boundary='Maximum rank-local duration per complete operator trial. No cross-GPU absolute timestamp alignment.',
        inner_repetitions=16),answer

@torch.inference_mode()
def main():
    mask=validate_mask();P=len(mask);assert P in [2,4]
    rank=int(os.environ['LOCAL_RANK']);torch.cuda.set_device(rank)
    dist.init_process_group('nccl',device_id=torch.device('cuda',rank))
    assert dist.get_world_size()==P
    assert str(torch.cuda.get_device_properties(rank).uuid).removeprefix('GPU-')==UUIDS[GPUS.index(mask[rank])]
    output=Path(os.environ['LOOPED_KV_OUTPUT']);records=[]
    for N in [3072,4096,8192,65536,98304,131072,786432,1572864,6291456,67108864]:
        payload=torch.zeros(N//2,device='cuda',dtype=torch.bfloat16)
        for primitive in ['broadcast','all_reduce']:
            def operate():
                if primitive=='broadcast':dist.broadcast(payload,src=0)
                else:dist.all_reduce(payload)
                return payload
            timing,value=timed(operate)
            assert torch.count_nonzero(value)==0
            records.append(dict(primitive=primitive,payload_bytes=N,P=P,physical_gpus=mask,
                timing=timing,label='GPU_MICROBENCH',
                traffic_scope='Tensor payload size, not a measured link-byte counter.',
                correctness='Zero-valued transport fixture stays exactly zero. No model approximation.'))
            if rank==0:print('COMMUNICATION',P,primitive,N,timing['median_ms'],flush=True)
        del payload
    for N in [4096,131072]:
        base=torch.arange(N//2,device='cuda',dtype=torch.int16)
        fresh=base.bitwise_xor(17);delta=torch.empty_like(base)
        packed=torch.empty((2,N//2),device='cuda',dtype=torch.uint8)
        restored=torch.empty_like(base)
        def transform():
            torch.bitwise_xor(base,fresh,out=delta)
            packed.copy_(delta.view(torch.uint8).reshape(-1,2).T)
            restored.copy_(packed.T.contiguous().reshape(-1).view(torch.int16))
            torch.bitwise_xor(base,restored,out=restored)
            return restored
        timing,value=timed(transform)
        assert torch.equal(value,fresh)
        records.append(dict(primitive='XOR_byteplane_roundtrip_no_compression',payload_bytes=N,P=P,
            physical_gpus=mask,timing=timing,bitwise_reconstruction=True,label='GPU_MICROBENCH',
            excluded='No entropy encoder, decoder, metadata handling, or transport. These require additional work.'))
    if rank==0:write(output/'communication.json',dict(records=records,P=P,physical_gpus=mask,
        UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),torch=torch.__version__,CUDA=torch.version.cuda,
        NCCL=torch.cuda.nccl.version(),label='GPU_MICROBENCH. Not actual branch runtime or a lossless GPU codec.',
        count_rule='One maximum for each full primitive. Do not sum component maxima as an observed request path.',
        bandwidth='Pure payload-movement interval is unresolved. No true link-bandwidth claim.'))
    dist.destroy_process_group()

if __name__=='__main__':main()
