"""Same native cache-free input, exact placement permutation, real four-rank EP.

All reference computation precedes sharding. No full reference expert weights
remain resident on GPUs during the EP measurements. Whole-prefix forwards are
not presented as cached serving TPOT.
"""
import argparse
import datetime
import gzip
import json
import os
import socket
import time
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from poc_looped_moe_ep.contract import *
from poc_looped_moe_ep.runtime import load_model,PhysicalEP


def greedy(model,prompt,eos,n=4):
    current=prompt;out=[]
    for _ in range(n):
        token=model(input_ids=current).logits[:,-1].argmax(-1)
        value=int(token[0]);out.append(value)
        if value==eos:break
        current=torch.cat((current,token[:,None]),dim=1)
    return out


def worker(rank,port,args):
    assert os.environ['CUDA_VISIBLE_DEVICES']==MASK
    torch.cuda.set_device(rank);torch.set_num_threads(2)
    assert str(torch.cuda.get_device_properties(rank).uuid).removeprefix('GPU-')==UUIDS[rank]
    torch.use_deterministic_algorithms(True);torch.utils.deterministic.fill_uninitialized_memory=False
    dist.init_process_group('nccl',init_method=f'tcp://127.0.0.1:{port}',rank=rank,world_size=4,
                            timeout=datetime.timedelta(seconds=180))
    model,tokenizer,snapshot=load_model(torch.device('cuda',rank))
    manifest=json.loads((ROOT/'results/frozen_requests.json').read_text())
    requests=[r for r in manifest['requests'] if r['split']==args.split]
    requests=requests[:args.limit] if args.limit else requests
    reference={};input_cache={}
    with torch.inference_mode():
        for request in requests:
            key=request['request_id'];base=request['prompt_tokens'];targets=request['target_tokens']
            prompt=torch.tensor([base],device='cuda');input_cache[key]=[]
            ref=dict(greedy=greedy(model,prompt,tokenizer.eos_token_id),last_logits=[],routes=[])
            for offset in [0,4,8]:
                tokens=torch.tensor([base+targets[:offset]],device='cuda');input_cache[key].append(tokens)
                sink={};native=model(input_ids=tokens,trace_sink=sink)
                ref['last_logits'].append(native.logits[0,-1].cpu())
                ref['routes'].append({(point.block,point.loop_step):dict(
                    ids=point.topk_idx[0].cpu(),weights=point.topk_weights[0].cpu(),
                    residual_sample=point.residual[0,-1,:32].cpu()) for point in sink.values()})
            reference[key]=ref
        core_placement=None
        if args.placement=='S1':
            core_placement=json.loads((ROOT/'results/frozen_best_static.json').read_text())['placement']
        ep=PhysicalEP(model,rank,core_placement)
        write(Path(args.output)/f'ownership.rank{rank}.json',ep.ownership)
        for index,request in enumerate(requests):
            placements=(['S0','S1'] if index%2==0 else ['S1','S0']) if args.placement=='both' else [args.placement]
            for placement in placements:
                if args.placement=='both':
                    mapping=([e//16 for e in range(64)] if placement=='S0' else
                             json.loads((ROOT/'results/frozen_best_static.json').read_text())['placement'])
                    ep.set_core_placement(mapping)
                    if index==0:write(Path(args.output)/f'ownership.{placement}.rank{rank}.json',ep.ownership)
                key=request['request_id'];prompts=input_cache[key]
                tag=key+'_'+placement if args.placement=='both' else key
                # Warmup is bounded and outside headline time.
                ep.mode='clean'
                for _ in range(2):
                    for tokens in prompts:model(input_ids=tokens)
                if index%2==0:order=['clean','timing','timing','clean','clean','timing','trace']
                else:order=['timing','clean','clean','timing','timing','clean','trace']
                for run_index,mode in enumerate(order):
                    ep.mode=mode;ep.records=[]
                    dist.barrier();torch.cuda.synchronize();start=time.perf_counter()
                    logits=[];physical_sinks=[]
                    for forward,tokens in enumerate(prompts):
                        ep.forward=forward;sink={} if mode=='trace' else None
                        logits.append(model(input_ids=tokens,trace_sink=sink).logits[0,-1])
                        physical_sinks.append(sink)
                    torch.cuda.synchronize();wall=(time.perf_counter()-start)*1000
                    # No CPU transfer, event resolution or cross-rank barrier inside wall.
                    local_records=ep.resolved() if mode!='clean' else []
                    if mode=='trace':
                        for row in local_records:
                            pool='loop' if row['pool']=='core' else row['pool']
                            native=reference[key]['routes'][row['forward']][(pool,row['loop'])]
                            row['native_router_ids_equal']=torch.equal(torch.tensor(row['ids']),native['ids'][rank::4])
                            row['native_router_weights_equal']=torch.equal(torch.tensor(row['weights']),native['weights'][rank::4].float())
                            point=next(point for point in physical_sinks[row['forward']].values()
                                       if (point.block,point.loop_step)==(pool,row['loop']))
                            row['native_residual_sample_equal']=torch.equal(
                                point.residual[0,-1,:32].cpu(),native['residual_sample'])
                    errors=[]
                    for expected,actual in zip(reference[key]['last_logits'],logits):
                        actual=actual.cpu();delta=actual.float()-expected.float()
                        errors.append(dict(exact=torch.equal(actual,expected),
                            rel_l2=float(delta.norm()/expected.float().norm().clamp_min(1e-9)),
                            max_scaled_abs=float(delta.abs().max()/expected.float().abs().max().clamp_min(1e-9)),
                            argmax_equal=int(actual.argmax())==int(expected.argmax())))
                    pass_all=all(e['rel_l2']<=.02 and e['max_scaled_abs']<=.04 and e['argmax_equal'] for e in errors)
                    if mode=='trace':pass_all &= all(row['native_router_ids_equal'] and
                        row['native_router_weights_equal'] and row['native_residual_sample_equal'] for row in local_records)
                    result=dict(request_id=key,split=args.split,placement=placement,rank=rank,
                        prompt_length=request['prompt_length'],prefix_offsets=[0,4,8],
                        mode=mode,run_index=run_index,wall_ms=wall,logit_errors=errors,pass_all=pass_all,
                        regime='B1 native H16 full-prefix next-token forwards, NOT cached decode TPOT')
                    write(Path(args.output)/f'{tag}_{mode}_{run_index}.rank{rank}.json',result)
                    if local_records:
                        with gzip.open(Path(args.output)/f'{tag}_{mode}_{run_index}.rank{rank}.events.json.gz','wt') as out:
                            json.dump(local_records,out)
                    assert pass_all,(key,mode,errors)
                ep.mode='clean'
                actual_greedy=greedy(model,prompts[0],tokenizer.eos_token_id)
                equality=actual_greedy==reference[key]['greedy']
                write(Path(args.output)/f'{tag}.rank{rank}.greedy.json',dict(
                    expected=reference[key]['greedy'],actual=actual_greedy,pass_all=equality))
                assert equality,(key,actual_greedy,reference[key]['greedy'])
                if rank==0:print('REQUEST_PASS',args.split,placement,index,key,flush=True)
    dist.barrier();dist.destroy_process_group()


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',required=True)
    p.add_argument('--split',choices=['calibration','held_out'],required=True)
    p.add_argument('--placement',choices=['S0','S1','both'],default='S0')
    p.add_argument('--limit',type=int);a=p.parse_args()
    output=Path(a.output);output.mkdir(parents=True,exist_ok=True)
    with socket.socket() as sock:sock.bind(('127.0.0.1',0));port=sock.getsockname()[1]
    mp.spawn(worker,args=(port,a),nprocs=4,join=True)
    write(output/'status.json',dict(success=True,split=a.split,placement=a.placement))


if __name__=='__main__':main()
