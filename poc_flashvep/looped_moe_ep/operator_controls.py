"""Real-weight expert scaling and exact independent-payload overlap controls."""
import time
import torch
import torch.distributed as dist
from .common import append

def replica_allocation(a,ep,target):
    """Allocate exact BF16 copies once. This is a memory/copy control, not a policy."""
    import hashlib
    import gc
    records=[]
    for budget in [1,ep.world]:
        retained=[]
        torch.cuda.synchronize()
        before=torch.cuda.memory_allocated()
        start=time.perf_counter()
        copies=[]
        for site in sorted(a.recurrent):
            block=a.sites[site]
            info=next(r for r in ep.ownership if r['site']==site)
            experts=info['logical_experts']
            for expert_id in range(min(budget,experts)):
                owner=expert_id%ep.world
                destination=(owner+1)%ep.world
                tensors=[]
                if a.name.startswith('ifm'):
                    expert=block.experts
                    all_values=[expert.weight1.reshape(-1,expert.inter_dim,expert.model_dim),
                        expert.weight2.reshape(-1,expert.model_dim,expert.inter_dim),
                        expert.weight3.reshape(-1,expert.inter_dim,expert.model_dim)]
                else:
                    all_values=[block.experts_w1,block.experts_w2,block.experts_w3]
                hashes=[]
                for values in all_values:
                    tensor=values[expert_id//ep.world].detach() if ep.rank==owner else torch.empty(values.shape[1:],device='cuda',dtype=torch.bfloat16)
                    dist.broadcast(tensor,src=owner)
                    hashes.append(hashlib.sha256(tensor.contiguous().view(torch.uint8).cpu().numpy().tobytes()).hexdigest())
                    if ep.rank==destination:
                        retained.append(tensor)
                    tensors.append(tensor.numel()*tensor.element_size())
                    del tensor
                copies.append(dict(site=site,expert=expert_id,source=owner,destination=destination,
                    bytes=sum(tensors),sha256=hashes))
        torch.cuda.synchronize()
        seconds=time.perf_counter()-start
        after=torch.cuda.memory_allocated()
        hashes_all=[None]*ep.world
        dist.all_gather_object(hashes_all,[r['sha256'] for r in copies])
        assert all(row==hashes_all[0] for row in hashes_all),'Replica value mismatch'
        records.append(dict(model=a.name,ep=ep.world,rank=ep.rank,extra_copies_per_recurrent_site=budget,
            retained_tensor_bytes=sum(x.numel()*x.element_size() for x in retained),
            allocated_delta_bytes=after-before,initialization_seconds=seconds,exact_replica_hashes=True,
            copies=copies,evidence='LIVE_ONCE_ONLY_REPLICA_ALLOCATION_AND_COPY_CONTROL_NOT_E2E'))
        del retained,all_values,values
        gc.collect()
        torch.cuda.empty_cache()
    for r in records:append(target/f'replica_memory.rank{ep.rank}.jsonl',r)

def measured(fn,rep=20,warm=10):
    for _ in range(warm):fn()
    values=[]
    for _ in range(rep):
        torch.cuda.synchronize()
        start=time.perf_counter()
        out=fn()
        torch.cuda.synchronize()
        values.append(1000*(time.perf_counter()-start))
    return values

def micro(a,ep,target,b,length):
    samples=ep.captured[-2:]
    assert samples
    site=ep.capture_site
    sample=samples[-1]
    x=sample['x']
    counts=sample['counts']
    # Rebatch within each identical expert. This measures shape scaling, not readiness.
    pieces=list(x.split(counts))
    for factor in [1,2,4]:
        joined=torch.cat([piece.repeat(factor,1) for piece in pieces],dim=0)
        joined_counts=[n*factor for n in counts]
        call=lambda:ep.expert_fns[site](joined,joined_counts) if len(joined) else joined
        values=measured(call)
        append(target/f'operator.rank{ep.rank}.jsonl',dict(model=a.name,ep=ep.world,rank=ep.rank,
            site=site,batch=b,input_tokens=length,source_loop=sample['loop'],factor=factor,
            counts=joined_counts,ms=values,rows=len(joined),
            evidence='LIVE_NATIVE_WEIGHT_OPERATOR_ONLY',
            excludes='Queue wait, pack, dispatch, combine, restoration and actual independent-request readiness.'))
    if b==1 and length==128:
        send=sample['send']
        recv=torch.empty(sample['receive_shape'],device=x.device,dtype=x.dtype)
        compute=torch.cuda.Stream()
        comm=torch.cuda.Stream()
        default=torch.cuda.current_stream()
        compute.wait_stream(default)
        comm.wait_stream(default)
        def collective():
            dist.all_to_all_single(recv,send,sample['recv_splits'],sample['send_splits'])
        def sequential():
            collective()
            return ep.expert_fns[site](x,counts) if len(x) else x
        def overlap():
            with torch.cuda.stream(comm):
                collective()
            with torch.cuda.stream(compute):
                out=ep.expert_fns[site](x,counts) if len(x) else x
            default.wait_stream(comm)
            default.wait_stream(compute)
            return out
        communication_values=measured(collective)
        sequential_values=measured(sequential)
        parallel_values=measured(overlap)
        reference=sequential().clone()
        actual=overlap()
        torch.cuda.synchronize()
        exact=torch.equal(reference,actual)
        assert exact,'Operator overlap changes the expert output'
        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,torch.profiler.ProfilerActivity.CUDA]) as profiler:
            with torch.profiler.record_function('INDEPENDENT_READY_DISPATCH_AND_EXPERT_OPERATOR'):
                overlap()
                torch.cuda.synchronize()
        profiler.export_chrome_trace(str(target/f'overlap_trace.rank{ep.rank}.json.gz'))
        append(target/f'overlap.rank{ep.rank}.jsonl',dict(model=a.name,ep=ep.world,rank=ep.rank,
            sequential_ms=sequential_values,overlap_ms=parallel_values,
            communication_ms=communication_values,
            expert_output_exact=exact,
            send_rows=len(send),recv_rows=len(recv),source_loop=sample['loop'],
            evidence='LIVE_INDEPENDENT_READY_PAYLOAD_OPERATOR_CONTROL_NOT_E2E',
            limitation='Two independent operator payloads from native tensors. Does not demonstrate live asynchronous request progress.'))
