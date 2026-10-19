"""Real P2/P4 operator measurements with pretrained weights and captured inputs.

These are component and ideal-ready-frontier tests, not live Ouro generation.
"""
import argparse
import ctypes
import datetime
import statistics
import time
from poc_ouro_parallelism_deep_discovery.runtime import configure_imports
from poc_ouro_parallelism_deep_discovery.contract import *
configure_imports()
import torch
import torch.distributed as dist
import torch.nn.functional as F
from safetensors import safe_open
USE_GRAPHS=False

def sync():torch.cuda.synchronize()

def timed(fn,n=31,warmup=7):
    for _ in range(warmup):fn()
    sync();dist.barrier();sync()
    inner=16 if USE_GRAPHS else 1
    graph=None
    if USE_GRAPHS:
        graph=torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            for _ in range(inner):captured=fn()
        graph.replay();sync();dist.barrier()
    times=[];events=[]
    for _ in range(n):
        dist.barrier();sync()
        a,b=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
        start=time.perf_counter();a.record()
        if graph is None:value=fn()
        else:graph.replay();value=captured
        b.record();sync()
        times.append((time.perf_counter()-start)*1000/inner)
        events.append(a.elapsed_time(b)/inner)
    gathered=[None]*dist.get_world_size()
    dist.all_gather_object(gathered,{'wall_ms':times,'stream_ms':events})
    # For this single-operator test, use each repeat's slowest participating rank.
    # No cross-component maximum is added to create a fabricated path.
    critical=[max(row['wall_ms'][i] for row in gathered) for i in range(n)]
    critical_stream=[max(row['stream_ms'][i] for row in gathered) for i in range(n)]
    return dict(samples_ms=critical,median_ms=statistics.median(critical),per_rank=gathered,
        critical_stream_ms=critical_stream,median_critical_stream_ms=statistics.median(critical_stream),
        graph=USE_GRAPHS,inner_repetitions=inner,
        residency='Repeated same operator and inputs. Warm-resident microbenchmark, not full-stack reuse distance.',
        boundary='Barrier before trial, local start through local GPU completion. Critical rank per full operator trial.',
        evidence_class='GPU_MICROBENCH'),value

@torch.inference_mode()
def main():
    p=argparse.ArgumentParser();p.add_argument('--size',choices=MODELS,required=True)
    p.add_argument('--capture',type=Path,required=True);p.add_argument('--port',type=int,default=29741)
    p.add_argument('--graphs',action='store_true')
    args=p.parse_args();mask=validate_mask();P=len(mask)
    global USE_GRAPHS
    USE_GRAPHS=args.graphs
    assert P in [2,4]
    rank=int(os.environ['LOCAL_RANK']);torch.cuda.set_device(rank)
    dist.init_process_group('nccl',device_id=torch.device('cuda',rank))
    assert dist.get_world_size()==P
    peer_group=dist.new_group(ranks=[0,1])
    uuid=str(torch.cuda.get_device_properties(rank).uuid).removeprefix('GPU-')
    assert uuid==UUIDS[GPUS.index(mask[rank])]
    out=Path(os.environ['LOOPED_KV_OUTPUT'])
    captured=torch.load(args.capture,map_location='cpu',weights_only=True)
    selected=next(value for key,value in captured.items() if '_layer0_call' in key and value.shape[0]>=4)
    info=MODELS[args.size]
    with safe_open(info['path']/'model.safetensors',framework='pt',device='cpu') as ck:
        full=[ck.get_tensor('model.layers.0.mlp.'+name+'.weight').to(device=rank,dtype=torch.bfloat16)
              for name in ['gate_proj','up_proj','down_proj']]
    wg,wu,wd=full;I=wg.shape[0]
    assert I%P==0
    part=[wg.chunk(P,dim=0)[rank].contiguous(),wu.chunk(P,dim=0)[rank].contiguous(),wd.chunk(P,dim=1)[rank].contiguous()]
    mapping=[None]*P
    properties=torch.cuda.get_device_properties(rank)
    dist.all_gather_object(mapping,{'logical_rank':rank,'physical_gpu':mask[rank],'UUID':uuid,
        'name':properties.name,'SM_count':properties.multi_processor_count,
        'total_memory_bytes':properties.total_memory,'L2_bytes':getattr(properties,'L2_cache_size',None)})
    results=[]
    for rows in [1,2,4,8,16,32,128,512]:
        x=selected.repeat((rows+selected.shape[0]-1)//selected.shape[0],1)[:rows].to(rank)
        def local():return F.linear(F.silu(F.linear(x,wg))*F.linear(x,wu),wd)
        def tp():
            result=F.linear(F.silu(F.linear(x,part[0]))*F.linear(x,part[1]),part[2])
            dist.all_reduce(result)
            return result
        native,reference=timed(local)
        sharded,observed=timed(tp)
        diff=observed.float()-reference.float()
        close=dict(max_abs=float(diff.abs().max()),relative_L2=float(diff.norm()/reference.float().norm().clamp_min(1e-12)),
            explanation='Numerically reordered TP sum. Not a full-model quality or token-parity result.')
        # Full replica per ready branch is a generic DP control, not loop-parallel execution.
        groups=max(P,rows)
        whole=selected.repeat((groups+selected.shape[0]-1)//selected.shape[0],1)[:groups].to(rank)
        branch_rows=whole[rank::P].contiguous()
        def branch_split():return F.linear(F.silu(F.linear(branch_rows,wg))*F.linear(branch_rows,wu),wd)
        branch_timing,_=timed(branch_split)
        combined,_=timed(lambda:F.linear(F.silu(F.linear(whole,wg))*F.linear(whole,wu),wd))
        # Source0 -> remote1 -> source0. Remote computation uses identical exact weights.
        send=torch.empty_like(x); back=torch.empty_like(x)
        def remote():
            if USE_GRAPHS:
                if rank<2:
                    if rank==0:send.copy_(x)
                    dist.broadcast(send,src=0,group=peer_group)
                    if rank==1:back.copy_(F.linear(F.silu(F.linear(send,wg))*F.linear(send,wu),wd))
                    dist.broadcast(back,src=1,group=peer_group)
                    return back
                return x
            if rank==0:
                dist.send(x,dst=1,group=peer_group);dist.recv(back,src=1,group=peer_group)
                return back
            if rank==1:
                dist.recv(send,src=0,group=peer_group)
                value=F.linear(F.silu(F.linear(send,wg))*F.linear(send,wu),wd)
                dist.send(value,dst=0,group=peer_group)
                return value
            return x
        remote_timing,returned=timed(remote)
        exact_remote=torch.equal(returned,reference) if rank in [0,1] else True
        if rank==0:
            results.append(dict(rows=rows,model=info['id'],P=P,local=native,TP=sharded,
                remote_round_trip=remote_timing,remote_operator_exact=exact_remote,numerical_TP=close,
                ready_branch_rows=groups,replica_split=branch_timing,one_worker_combined=combined,
                activation_bytes_one_way=rows*2048*2,full_MLP_weight_bytes=sum(w.numel()*w.element_size() for w in full),
                local_vs_TP_speedup=native['median_ms']/sharded['median_ms'],
                local_vs_remote_speedup=native['median_ms']/remote_timing['median_ms'],
                remote_primitive='NCCL graph-captured peer broadcasts' if USE_GRAPHS else 'NCCL send/receive',
                ready_frontier_scope='Independent captured inputs supplied in advance. NOT a legal same-token future-depth schedule.',
                source_capture=str(args.capture),source_capture_sha256=sha(args.capture),evidence_class='GPU_MICROBENCH'))
            print('COMPONENT',args.size,P,rows,results[-1]['local_vs_remote_speedup'],flush=True)
            write(out/'components.json',dict(results=results,mapping=mapping,
                UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                checkpoint_revision=info['revision'],torch=torch.__version__,CUDA=torch.version.cuda,
                NCCL=torch.cuda.nccl.version(),
                precision='BF16. No changed expert, gate, routing, or model weights.',
                label='GPU_MICROBENCH. No full-model speedup or quality claim.'))
    dist.destroy_process_group()

if __name__=='__main__':main()
