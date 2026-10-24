"""Real NCCL context-sharded attention over original model-derived BF16 KV."""
import argparse
import gc
import hashlib
import json
import os
import time
from datetime import timedelta
from pathlib import Path
import numpy as np
import torch
import torch.distributed as dist
from .common import ROOT,FIXTURES,UUIDS,write,digest
PRESSURE=None

def tensor_hash(t):
    return hashlib.sha256(t.contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()

def paged(k,v,batch):
    length=k.shape[0];pages=(length+15)//16
    ks=torch.zeros((batch*pages,16,16,128),device='cuda',dtype=torch.bfloat16)
    vs=torch.zeros_like(ks)
    for b in range(batch):
        ks[b*pages:(b+1)*pages].view(-1,16,128)[:length].copy_(k)
        vs[b*pages:(b+1)*pages].view(-1,16,128)[:length].copy_(v)
    table=torch.arange(batch*pages,device='cuda',dtype=torch.int32).view(batch,pages)
    lengths=torch.full((batch,),length,device='cuda',dtype=torch.int32)
    return ks,vs,table,lengths

def attention(q,cache,return_lse=True,num_splits=1):
    from flash_attn.cute import flash_attn_varlen_func
    k,v,table,lengths=cache
    return flash_attn_varlen_func(q.unsqueeze(1),k,v,page_table=table,seqused_k=lengths,
        max_seqlen_q=1,max_seqlen_k=table.shape[1]*16,causal=False,num_splits=num_splits,return_lse=return_lse)

def bench(function,repeats=25):
    for _ in range(5):function()
    torch.cuda.synchronize()
    rows=[];gpu=[]
    for _ in range(repeats):
        if PRESSURE is not None:
            PRESSURE.add_(1)
            torch.cuda.synchronize()
        start=torch.cuda.Event(enable_timing=True);end=torch.cuda.Event(enable_timing=True)
        wall=time.perf_counter();start.record();function();end.record();end.synchronize()
        rows.append((time.perf_counter()-wall)*1000);gpu.append(start.elapsed_time(end))
    return dict(wall_ms=rows,cuda_ms=gpu,wall_median_ms=float(np.median(rows)),cuda_median_ms=float(np.median(gpu)))

def graph(function):
    stream=torch.cuda.Stream();stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):result=function()
    torch.cuda.current_stream().wait_stream(stream);torch.cuda.synchronize()
    g=torch.cuda.CUDAGraph()
    with torch.cuda.graph(g,stream=stream):result=function()
    return g,result

def metrics(a,b):
    a=a.float();b=b.float()
    return dict(relative_L2=float(torch.linalg.vector_norm(a-b)/torch.linalg.vector_norm(a).clamp_min(1e-12)),
        absolute_max=float((a-b).abs().max()),finite=bool(torch.isfinite(b).all()))

@torch.inference_mode()
def main():
    global PRESSURE
    p=argparse.ArgumentParser();p.add_argument('--run',required=True);p.add_argument('--fixtures',required=True)
    p.add_argument('--batches',default='1,2,4,8');p.add_argument('--repeats',type=int,default=25)
    p.add_argument('--skew',action='store_true');p.add_argument('--cache-pressure',action='store_true')
    p.add_argument('--components',action='store_true');p.add_argument('--profile',action='store_true');args=p.parse_args()
    rank=int(os.environ['RANK']);world=int(os.environ['WORLD_SIZE']);local=int(os.environ['LOCAL_RANK'])
    assert world in (2,4,8) and torch.cuda.device_count()==world
    torch.cuda.set_device(local);torch.set_num_threads(1)
    devices=list(map(int,os.environ['CUDA_VISIBLE_DEVICES'].split(',')))
    assert devices==list(range(world))
    uuid=str(torch.cuda.get_device_properties(local).uuid)
    assert uuid.removeprefix('GPU-')==UUIDS[devices[local]].removeprefix('GPU-')
    dist.init_process_group('nccl',timeout=timedelta(minutes=15),device_id=torch.device('cuda',local))
    if args.cache_pressure:PRESSURE=torch.zeros(32*1024*1024,device='cuda',dtype=torch.float32)
    target=ROOT/'distributed'/args.run;target.mkdir(parents=True,exist_ok=True)
    identity=dict(rank=rank,pid=os.getpid(),physical_gpu=devices[local],uuid=uuid,world=world,
        backend=dist.get_backend(),peer_access={str(j):torch.cuda.can_device_access_peer(local,j) for j in range(world) if j!=local},
        source_capture_wall_s=time.time(),experiment_source_sha256=digest(Path(__file__)))
    write(target/f'rank{rank}_identity.json',identity)
    write(target/f'rank{rank}_sources.json',{p.name:dict(sha256=digest(p),text=p.read_text()) for p in Path(__file__).parent.glob('*.py')})
    selected=json.loads(Path(args.fixtures).read_text());rows=[]
    if world==2 and (FIXTURES/'P32768_d1_l12.pt').exists():
        from .peer_measure import run as transfers
        transfers(target,rank,world)
    contract=json.loads((ROOT/'prospective_contract.json').read_text())['operator_tolerance']
    for source in selected:
        fixture=FIXTURES/source['name'];assert digest(fixture)==source['sha256']
        packet=torch.load(fixture,map_location='cpu',weights_only=True)
        assert all(packet[key].dtype==torch.bfloat16 for key in ('q','k','v'))
        if 'length_override' in source:
            packet['k']=packet['k'][:source['length_override']]
            packet['v']=packet['v'][:source['length_override']]
        length=packet['k'].shape[0]
        for batch in map(int,args.batches.split(',')):
            modes=['balanced','skewed'] if args.skew else ['balanced']
            for mode in modes:
                if mode=='balanced':bounds=[length*i//world for i in range(world+1)]
                else:
                    first=length*3//4
                    bounds=[0,first]+[first+(length-first)*i//(world-1) for i in range(1,world)]
                lo,hi=bounds[rank:rank+2]
                kh=tensor_hash(packet['k'][lo:hi]);vh=tensor_hash(packet['v'][lo:hi])
                local_cache=paged(packet['k'][lo:hi],packet['v'][lo:hi],batch)
                # Check actual GPU shard payloads before attention.
                valid_k=local_cache[0][:((hi-lo+15)//16)].reshape(-1,16,128)[:hi-lo].cpu()
                valid_v=local_cache[1][:((hi-lo+15)//16)].reshape(-1,16,128)[:hi-lo].cpu()
                assert tensor_hash(valid_k)==kh and tensor_hash(valid_v)==vh
                q=packet['q'].repeat(batch,1,1).to(device='cuda')
                assert q.shape==(batch,16,128)
                reference=torch.empty_like(q)
                native={}
                if rank==0:
                    full=paged(packet['k'],packet['v'],batch)
                    local_function=lambda:attention(q,full,return_lse=False)[0].squeeze(1)
                    reference.copy_(local_function())
                    native['eager']=bench(local_function,args.repeats)
                    native_graph,native_out=graph(local_function)
                    native['graph']=bench(native_graph.replay,args.repeats)
                    native['local_split_controls']={}
                    splits_to_test=(2,4,8) if torch.cuda.get_device_capability()[0]!=9 else ()
                    if not splits_to_test:
                        native['local_split_controls']['UNSUPPORTED']='Pinned FA4 raises: SplitKV not supported on SM 9.0. See preserved scaling_v1 failure.'
                    for splits in splits_to_test:
                        f=lambda:attention(q,full,return_lse=False,num_splits=splits)[0].squeeze(1)
                        split_graph,split_out=graph(f)
                        control=bench(split_graph.replay,args.repeats)
                        control['numerical']=metrics(reference,split_out)
                        native['local_split_controls'][str(splits)]=control
                        del split_graph,split_out,f
                    # Baseline uses exactly the same original pretrained Q/K/V.
                    native['original_capture_difference']=metrics(packet['out'].to('cuda').repeat(batch,1,1),reference) if 'length_override' not in source else None
                    del native_graph,native_out,full,local_function
                dist.broadcast(reference,src=0)
                torch.cuda.synchronize();torch.cuda.empty_cache();dist.barrier()
                residency_before_timing=dict(allocated=torch.cuda.memory_allocated(),reserved=torch.cuda.memory_reserved())
                send=torch.empty((batch,16,129),device='cuda',dtype=torch.float32)
                recv=torch.empty((world*batch,16,129),device='cuda',dtype=torch.float32)
                def distributed_attention():
                    dist.broadcast(q,src=0)
                    if hi>lo:
                        partial,lse=attention(q,local_cache)
                        send[...,:128].copy_(partial.squeeze(1))
                        send[...,128].copy_(lse.reshape(batch,16))
                    else:
                        send[...,:128].zero_();send[...,128].fill_(-float('inf'))
                    dist.all_gather_into_tensor(recv,send)
                    summaries=recv.view(world,batch,16,129)
                    weights=torch.softmax(summaries[...,128],dim=0)
                    return (summaries[...,:128]*weights.unsqueeze(-1)).sum(0).to(torch.bfloat16)
                if rank!=0:q.zero_()
                actual=distributed_attention();torch.cuda.synchronize()
                assert torch.equal(q.cpu(),packet['q'].repeat(batch,1,1))
                numeric=metrics(reference,actual)
                assert numeric['finite'] and numeric['relative_L2']<=contract['relative_L2'] and numeric['absolute_max']<=contract['absolute_max'],numeric
                eager=bench(distributed_attention,args.repeats)
                graph_stats=None;graph_error=None
                components=None
                try:
                    dist.barrier();cg,out=graph(distributed_attention)
                    if args.profile:torch.cuda.cudart().cudaProfilerStart()
                    graph_stats=bench(cg.replay,args.repeats)
                    if args.profile:torch.cuda.cudart().cudaProfilerStop()
                    cg.replay();torch.cuda.synchronize()
                    graph_stats['numerical']=metrics(reference,out)
                    del cg,out
                except Exception as error:
                    graph_error=repr(error)
                if args.components:
                    partial=lambda:attention(q,local_cache)
                    pg,po=graph(partial);partial_stats=bench(pg.replay,args.repeats)
                    del pg,po,partial
                    def comm_merge():
                        dist.broadcast(q,src=0);dist.all_gather_into_tensor(recv,send)
                        summaries=recv.view(world,batch,16,129)
                        weights=torch.softmax(summaries[...,128],dim=0)
                        return (summaries[...,:128]*weights.unsqueeze(-1)).sum(0).to(torch.bfloat16)
                    mg,mo=graph(comm_merge);comm_stats=bench(mg.replay,args.repeats)
                    del mg,mo,comm_merge
                    components=dict(local_partial_graph=partial_stats,communication_merge_graph=comm_stats,
                        qualifier='Independent component measurements. Their sum is not a measured critical path.')
                row=dict(fixture=source['name'],fixture_sha256=source['sha256'],depth=packet['depth'],layer=packet['layer'],
                    split=packet['split'],L=length,B=batch,P=world,rank=rank,placement=mode,lo=lo,hi=hi,
                    owned_valid_KV_bytes=(hi-lo)*16*128*2*2*batch,
                    allocated_KV_bytes=sum(v.numel()*v.element_size() for v in local_cache[:2]),
                    original_K_sha256=kh,original_V_sha256=vh,numerical=numeric,
                    GPU_shard_byte_exact=True,
                    query_owner_broadcast_validated=True,residency_before_timing=residency_before_timing,
                    eager=eager,graph=graph_stats,graph_error=graph_error,native=native,
                    query_payload_bytes=q.numel()*2,summary_payload_bytes=send.numel()*4,
                    collectives_per_attention=dict(broadcast=1,all_gather=1),
                    communication='REAL NCCL broadcast Q and all-gather output/LSE. No full KV gather in timed sharded path.',
                    evidence='LIVE_MEASURED_OPERATOR',batch_provenance='B copies of one real model query/KV. Operator geometry, not independent serving requests.',
                    causal_semantics='One query over the explicitly valid causal prefix. Local causal=False retains every supplied valid key once.',
                    edge_prefix_override=source.get('length_override'))
                row['cache_pressure_bytes']=PRESSURE.numel()*PRESSURE.element_size() if PRESSURE is not None else 0
                row['components']=components;row['heavy_profile']=args.profile
                rows.append(row)
                if len(rows)%8==0:write(target/f'rank{rank}_results.json',rows)
                dist.barrier()
                del local_cache,q,reference,send,recv,actual,distributed_attention
        del packet
        gc.collect();torch.cuda.empty_cache()
        write(target/f'rank{rank}_results.json',rows)
    write(target/f'rank{rank}_completed.json',dict(rows=len(rows),end_wall_s=time.time()))
    dist.barrier();dist.destroy_process_group()

if __name__=='__main__':main()
