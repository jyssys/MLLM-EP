"""Physical P2P, interference, and conventional exact CP operator surrogates."""
import gc
import importlib
import os
import statistics
import time
import torch
import numpy as np
from poc_recurrent_asymmetry.contract import *
from poc_async_recurrent_pp.scripts.baseline import load


def sync_all():
    for device in range(4):torch.cuda.synchronize(device)


def measure(function,repeats=30,warmup=5):
    for _ in range(warmup):function();sync_all()
    values=[]
    for _ in range(repeats):
        sync_all();start=time.perf_counter();result=function();sync_all()
        values.append((time.perf_counter()-start)*1000)
    return dict(wall_ms=values,median_ms=statistics.median(values),p95_ms=sorted(values)[int(.95*(len(values)-1))],
        repeats=repeats,warmup=warmup,boundary='One host clock, all participating GPU completion. Includes enqueue, allocation, and synchronization.'),result


def flash(q,k,v):
    result=torch.ops.aten._scaled_dot_product_flash_attention(q,k,v,0.,False,False,scale=128**-.5)
    return result[0],result[1]


@torch.inference_mode()
def main():
    assert os.environ['CUDA_VISIBLE_DEVICES']==MASK and torch.cuda.device_count()==4
    output=Path(os.environ['LOOPED_KV_OUTPUT'])
    mapping=[]
    for i in range(4):
        prop=torch.cuda.get_device_properties(i)
        assert str(prop.uuid).removeprefix('GPU-')==UUIDS[i]
        mapping.append(dict(logical_rank=i,physical_gpu=GPUS[i],UUID=str(prop.uuid),name=prop.name,
            HBM_bytes=prop.total_memory,peer_access=[torch.cuda.can_device_access_peer(i,j) if i!=j else True for j in range(4)]))
    write(output/'mapping.json',mapping)
    transfers=[]
    sizes=sorted(set([r*2048*2 for r in [1,32,128,256,512,1024]]+[32*1024**2,128*1024**2,256*1024**2]))
    for source in range(4):
        for destination in range(4):
            if source==destination:continue
            for size in sizes:
                x=torch.ones(size//2,device=f'cuda:{source}',dtype=torch.bfloat16)*(source+1)
                y=torch.empty_like(x,device=f'cuda:{destination}')
                with torch.cuda.device(destination):
                    stream=torch.cuda.Stream();gpu=[]
                    def copy_once():
                        a=torch.cuda.Event(enable_timing=True);b=torch.cuda.Event(enable_timing=True)
                        with torch.cuda.stream(stream):
                            a.record();y.copy_(x,non_blocking=True);b.record()
                        gpu.append((a,b))
                    timing,_=measure(copy_once)
                    assert bool((y==source+1).all())
                    gpu_times=[a.elapsed_time(b) for a,b in gpu[-30:]]
                transfers.append(dict(source=source,destination=destination,payload_bytes=size,
                    **timing,destination_stream_ms=gpu_times,
                    apparent_GB_per_s=size/(statistics.median(gpu_times)*1e6),
                    scope='Physical peer copy. Useful payload / stream interval is apparent throughput, not counter-measured link bandwidth.'))
                del x,y;gc.collect()
    write(output/'p2p.json',transfers)
    torch.cuda.set_device(0)
    model,tokenizer=load();manifest=read(ROOT/'metrics/frozen_workload.json')
    cp=[]
    for P in manifest['contexts']:
        ids=torch.tensor([manifest['requests'][0]['prompts'][str(P)]],device='cuda:0')
        value=model(ids,use_cache=True,exit_at_step=3,logits_to_keep=1)
        module=importlib.import_module(type(model).__module__);registry=module.ALL_ATTENTION_FUNCTIONS
        original=registry['sdpa'];captured={};counter=0
        def capture(attention,q,k,v,*args,**kwargs):
            nonlocal counter
            if attention.layer_idx==0:
                counter+=1
                if counter in [1,4]:captured[counter]=(q,k,v)
            return original(attention,q,k,v,*args,**kwargs)
        registry['sdpa']=capture
        try:updated=model(torch.tensor([[13]],device='cuda:0'),past_key_values=value.past_key_values,
            use_cache=True,exit_at_step=3,logits_to_keep=1)
        finally:registry['sdpa']=original
        for loop,(q,k,v) in captured.items():
            assert q.shape[2]==1 and k.shape[2]==P+1
            q=q.contiguous();k=k.contiguous();v=v.contiguous()
            native,reference=measure(lambda:flash(q,k,v)[0])
            boundaries=[(P+1)*i//4 for i in range(5)]
            shards=[];sync_all();start=time.perf_counter()
            for i in range(4):
                shards.append((k[:,:,boundaries[i]:boundaries[i+1]].contiguous().to(f'cuda:{i}'),
                               v[:,:,boundaries[i]:boundaries[i+1]].contiguous().to(f'cuda:{i}')))
            sync_all();placement_ms=(time.perf_counter()-start)*1000
            def distributed():
                partial=[]
                for i,(key,val) in enumerate(shards):
                    with torch.cuda.device(i):
                        query=q.to(f'cuda:{i}',non_blocking=True)
                        result,lse=flash(query,key,val)
                        partial.append((result,lse))
                with torch.cuda.device(0):
                    outputs=[part[0].to('cuda:0',non_blocking=True).float() for part in partial]
                    lses=[part[1].to('cuda:0',non_blocking=True).float() for part in partial]
                    maximum=torch.stack(lses).max(0).values
                    weights=[(s-maximum).exp() for s in lses]
                    numerator=sum(o*w.unsqueeze(-1) for o,w in zip(outputs,weights))
                    return (numerator/sum(weights).unsqueeze(-1)).to(q.dtype)
            timing,result=measure(distributed)
            difference=(result.float()-reference.float()).abs()
            relative=float(torch.linalg.vector_norm(difference)/torch.linalg.vector_norm(reference.float()).clamp_min(1e-12))
            # Frozen before measurement. BF16 partial reductions need not be bitwise identical.
            correct=float(difference.max())<=.02 and relative<=.02
            cp.append(dict(context=P,loop=loop,native=native,CP4=timing,
                correctness=correct,max_abs=float(difference.max()),relative_L2=relative,
                fixed_tolerance=dict(max_abs=.02,relative_L2=.02),KV_placement_ms=placement_ms,
                resident_KV_payload_bytes=k.numel()*k.element_size()*2,
                ranges=list(zip(boundaries,boundaries[1:])),current_position_included_once=True,
                merge='Local attention output plus LSE. Global max, exp weights, normalized weighted sum.',
                speedup=native['median_ms']/timing['median_ms'],
                label='MEASURED_CONVENTIONAL_CP_OPERATOR_MICROBENCH. NOT_OURO_E2E.',
                limitation='One-process four-GPU launches and temporary gather buffers. No optimized distributed engine or TP4×CP4 assumption.'))
            assert correct,'Distributed softmax operator correctness failed'
            del shards,boundaries
        write(output/f'cp_P{P}.json',[r for r in cp if r['context']==P])
        if P in [1024,16384]:
            logits=[value.logits[0,-1].float().cpu().numpy(),updated.logits[0,-1].float().cpu().numpy()]
            for _ in range(2):
                updated=model(torch.tensor([[13]],device='cuda:0'),past_key_values=updated.past_key_values,
                    use_cache=True,exit_at_step=3,logits_to_keep=1)
                logits.append(updated.logits[0,-1].float().cpu().numpy())
            positions=[0,1,P-1,P,P+2]
            coordinates=torch.tensor([0,1,2,3,124,125,126,127,128,129,130,131,252,253,254,255],device='cuda:0')
            samples=[]
            for key,val in zip(updated.past_key_values.key_cache,updated.past_key_values.value_cache):
                combined=torch.cat([key,val],dim=-1)
                samples.append(combined[0,:,positions,:].index_select(-1,coordinates).permute(1,0,2).float().cpu().numpy())
            path=output/f'native_reference_P{P}.npz'
            np.savez_compressed(path,logits=np.stack(logits),KV=np.stack(samples),positions=positions)
            write(output/f'native_reference_P{P}.json',dict(path=str(path),sha256=sha(path),
                slots=96,query_heads=16,KV_heads=16,prefix=P,teacher_after_prefill=[13,13,13],
                precision='BF16 forward, FP32 diagnostic copies',scope='Exact HF fidelity reference for a separately qualified native runtime.'))
        del value,updated,captured,ids,q,k,v,result,reference;gc.collect();torch.cuda.empty_cache()
        print('CP_OPERATOR',P,flush=True)
    write(output/'cp_operator.json',cp)
    overlap=[]
    x=torch.ones((1024,2048),device='cuda:0',dtype=torch.bfloat16)
    source=torch.ones(64*1024**2//2,device='cuda:1',dtype=torch.bfloat16)
    destination=torch.empty_like(source,device='cuda:0')
    compute=lambda:model.model.layers[0].mlp(x)
    solo_compute,expected=measure(compute)
    solo_copy,_=measure(lambda:destination.copy_(source,non_blocking=True))
    stream_a=torch.cuda.Stream(device=0);stream_b=torch.cuda.Stream(device=0)
    def concurrent():
        with torch.cuda.stream(stream_a):out=compute()
        with torch.cuda.stream(stream_b):destination.copy_(source,non_blocking=True)
        return out
    together,observed=measure(concurrent)
    assert torch.equal(expected,observed) and bool((destination==1).all())
    write(output/'overlap.json',dict(compute=solo_compute,copy=solo_copy,concurrent=together,
        serial_sum_ms=solo_compute['median_ms']+solo_copy['median_ms'],
        compute_shape=list(x.shape),copy_bytes=source.numel()*source.element_size(),
        scope='Real pretrained first-layer MLP and disjoint 64 MiB peer copy. Operator interference, not pipeline speedup.'))
    write(output/'status.json',dict(success=True,physical_GPUs=GPUS,CP_correct=all(r['correctness'] for r in cp),
        scope='Measured physical microbenchmarks only. No new inference engine.'))


if __name__=='__main__':main()
