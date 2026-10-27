"""Complete model and generation timing. Heavy traces run separately."""
import json
import time
import torch
import torch.distributed as dist
from .common import append,write
from .workloads import inputs
from .traces import RouterTrace

def references(a,config):
    result={}
    for b in config.get('batches',[1,2,4,8,16]):
        for length in config.get('lengths',[128,512,2048]):
            ids,requests=inputs(a,b,length)
            logits=a.forward(ids)
            result[(b,length)]=dict(last=logits[:,-1].detach().float().cpu(),top1=logits.argmax(-1).cpu())
            del logits
    torch.cuda.empty_cache()
    return result

def timed(fn):
    dist.barrier();torch.cuda.synchronize();start=time.perf_counter()
    out=fn();torch.cuda.synchronize();elapsed=(time.perf_counter()-start)*1000
    maximum=torch.tensor([elapsed],device='cuda',dtype=torch.float64);dist.all_reduce(maximum,op=dist.ReduceOp.MAX)
    return maximum.item(),out

def generate(a,ids,steps):
    prefix=ids.clone();token_ids=[];timestamps=[]
    dist.barrier();torch.cuda.synchronize();start=time.perf_counter()
    for _ in range(steps):
        logits=a.forward(prefix)
        token=logits[:,-1].argmax(-1,keepdim=True)
        torch.cuda.synchronize()
        token_ids.append(token.flatten().cpu().tolist())
        timestamps.append((time.perf_counter()-start)*1000)
        prefix=torch.cat([prefix,token],dim=-1)
        del logits
    return dict(e2e_ms=timestamps[-1],ttft_ms=timestamps[0],tpot_ms=(timestamps[-1]-timestamps[0])/(steps-1),
        timestamps_ms=timestamps,generated_ids=token_ids,cap=steps,eos_enabled=False,
        scope='LIVE_SYNCHRONIZED_HARDWARE_GENERATION_FULL_PREFIX_RECOMPUTE_NOT_NATURAL_ACCURACY'),prefix

def measure(a,ep,target,config,reference):
    for b in config.get('batches',[1,2,4,8,16]):
        for length in config.get('lengths',[128,512,2048]):
            ids,requests=inputs(a,b,length)
            ep.request_ids=[r['id'] for r in requests]
            logits=a.forward(ids)
            ref=reference[(b,length)]
            last=logits[:,-1].float().cpu()
            l2=float((last-ref['last']).norm()/ref['last'].norm().clamp_min(1e-9))
            top1_equal=int((logits.argmax(-1).cpu()==ref['top1']).sum())
            append(target/f'shape_fidelity.rank{ep.rank}.jsonl',dict(batch=b,prefix=length,last_logit_l2=l2,
                top1_equal=top1_equal,positions=ids.numel(),pass_fidelity=l2<=.02 and top1_equal/ids.numel()>=.99))
            assert l2<=.02 and top1_equal/ids.numel()>=.99
            del logits
            ep.captured=[]
            for _ in range(config.get('warmups',10)):
                out=a.forward(ids);del out
            torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats()
            clean=[]
            for _ in range(config.get('repeats',5)):
                duration,out=timed(lambda:a.forward(ids));clean.append(duration);del out
            peak=torch.cuda.max_memory_allocated()
            ep.trace=True
            traced,out=timed(lambda:a.forward(ids))
            events=ep.resolve();ep.trace=False;del out
            write(target/f'ep_trace_b{b}_p{length}.rank{ep.rank}.json',events)
            ep.captured=[]
            router=RouterTrace(a);router.request_ids=[r['id'] for r in requests];router.domain=[r['domain'] for r in requests]
            diagnostic,out=timed(lambda:a.forward(ids));del out
            router.close()
            if ep.rank==0:
                for row in router.records:append(target/'routes.jsonl',row)
            write(target/f'forward_b{b}_p{length}.rank{ep.rank}.json',dict(batch=b,prefix=length,clean_ms=clean,
                observer_ep_ms=traced,observer_router_ms=diagnostic,peak_allocated_bytes=peak,
                reserved_bytes=torch.cuda.memory_reserved(),allocated_bytes=torch.cuda.memory_allocated(),requests=[r['id'] for r in requests],
                evidence='LIVE_TRUE_EP_COMPLETE_FORWARD',token_work='All rows at every native recurrent pass'))
            cap=config.get('generation_cap',32)
            # One full warmup reaches every growing context shape before clean generation.
            warm,prefix=generate(a,ids,cap)
            for repeat in range(config.get('generation_repeats',3)):
                value,prefix=generate(a,ids,cap)
                assert value['generated_ids']==warm['generated_ids'],'Warmup changes deterministic hardware tokens'
                append(target/f'generation_b{b}_p{length}.rank{ep.rank}.jsonl',dict(value,batch=b,prefix=length,repeat=repeat,
                    output_tokens=b*cap,warmup_full_generations=1))
            # Representative final decode forward only. Never multiply this into an E2E breakdown.
            router=RouterTrace(a);router.request_ids=[r['id'] for r in requests];router.domain=[r['domain'] for r in requests]
            router.phase='decode';router.step=cap-1
            final_input=prefix[:,:-1].contiguous()
            ep.trace=True
            decode_observer,out=timed(lambda:a.forward(final_input));events=ep.resolve();ep.trace=False;del out
            router.close()
            write(target/f'ep_decode_b{b}_p{length}.rank{ep.rank}.json',events)
            ep.captured=[]
            if ep.rank==0:
                for row in router.records:append(target/'routes.jsonl',row)
            del prefix
            print('MEASURED',a.name,ep.world,b,length,sum(clean)/len(clean),flush=True)
            # Stronger generic control: current-router source ownership, no pass feature.
            ep.source_policy='source_owner'
            generic=a.forward(ids)
            last=generic[:,-1].float().cpu()
            relative=float((last-ref['last']).norm()/ref['last'].norm().clamp_min(1e-9))
            top1=int((generic.argmax(-1).cpu()==ref['top1']).sum())
            append(target/f'generic_fidelity.rank{ep.rank}.jsonl',dict(batch=b,prefix=length,last_logit_l2=relative,
                top1_equal=top1,positions=ids.numel(),pass_fidelity=relative<=.02 and top1/ids.numel()>=.99))
            assert relative<=.02 and top1/ids.numel()>=.99
            del generic
            for _ in range(config.get('warmups',10)):
                out=a.forward(ids);del out
            torch.cuda.reset_peak_memory_stats()
            generic_ms=[]
            for _ in range(config.get('repeats',5)):
                duration,out=timed(lambda:a.forward(ids));generic_ms.append(duration);del out
            generic_peak=torch.cuda.max_memory_allocated()
            ep.trace=True
            observer,out=timed(lambda:a.forward(ids));generic_events=ep.resolve();ep.trace=False;del out
            ep.captured=[]
            write(target/f'generic_trace_b{b}_p{length}.rank{ep.rank}.json',generic_events)
            write(target/f'generic_forward_b{b}_p{length}.rank{ep.rank}.json',dict(batch=b,prefix=length,clean_ms=generic_ms,
                observer_ms=observer,source_policy='Current top-1 owner, exact current router, no loop information',
                peak_allocated_bytes=generic_peak,allocated_bytes=torch.cuda.memory_allocated(),reserved_bytes=torch.cuda.memory_reserved(),
                state_movement_bytes=0,movement_scope='Existing replicated dense states only. Not an Attention-DP or KV-sharded claim.'))
            generic_warm,_=generate(a,ids,cap)
            for repeat in range(config.get('generation_repeats',3)):
                value,_=generate(a,ids,cap)
                append(target/f'generic_generation_b{b}_p{length}.rank{ep.rank}.jsonl',dict(value,batch=b,prefix=length,repeat=repeat,
                    output_tokens=b*cap,static_token_exact=value['generated_ids']==warm['generated_ids'],
                    generic_warm_token_exact=value['generated_ids']==generic_warm['generated_ids']))
                assert value['generated_ids']==generic_warm['generated_ids']
            ep.source_policy='static'
    # Vary B and context in one loaded model. No fixed-B dispatcher assumption.
    for b,length in [(2,128),(16,512),(1,2048),(8,128),(4,512)]:
        if b not in config.get('batches',[1,2,4,8,16]):continue
        ids,_=inputs(a,b,length);out=a.forward(ids)
        append(target/f'mixed_shapes.rank{ep.rank}.jsonl',dict(batch=b,prefix=length,finite=bool(torch.isfinite(out).all()),
            full_site_calls=dict(ep.calls)))
        del out
    if config.get('kernel_profile',len(config.get('batches',[]))==5):
        from .kernel_profile import profile
        try:profile(a,ep,target)
        except Exception as error:
            write(target/f'kernel_profile_blocked.rank{ep.rank}.json',dict(status='BLOCKED',error=repr(error),
                scope='Separate heavy observer. Clean measurements remain preserved.'))

