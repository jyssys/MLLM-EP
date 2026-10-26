"""Clean whole-model timings and separate actual per-rank trace measurements."""
import json
import time
import torch
import torch.distributed as dist
import torch.nn.functional as F
from .common import *

def corpus():
    return json.loads((PRIOR/'workload_manifest.json').read_text())['rows']

def shaped_inputs(a,b,length,offset=0):
    rows=[r for r in corpus() if r['dataset']=='language' and r['split']=='calibration']
    selected=[rows[(offset+i)%len(rows)] for i in range(b)]
    sequences=[]
    for row in selected:
        text=row['text']+'\n'
        ids=a.encode(text,length)
        while ids.shape[-1]<length:
            text=text+text
            ids=a.encode(text,length)
        sequences.append(ids[0])
    return torch.stack(sequences),[r['id'] for r in selected]

def prepare_references(a,count=32):
    records=[]
    for domain in ['language','gsm8k','math500']:
        rows=[r for r in corpus() if r['dataset']==domain and r['split']=='heldout'][:count]
        for row in rows:
            ids=a.encode(row['text'],128)
            logits=a.forward(ids)
            loss=F.cross_entropy(logits[:,:-1].float().reshape(-1,logits.shape[-1]),ids[:,1:].reshape(-1)).item()
            records.append(dict(id=row['id'],dataset=domain,ids=ids.cpu(),last_logits=logits[:,-1].float().cpu(),
                nll=loss,top1=logits.argmax(-1).cpu()))
    return records

def prepare_shape_references(a,config):
    records=[]
    for b in config.get('batches',[1,2,4]):
        for length in config.get('lengths',[128,1024,2048]):
            ids,requests=shaped_inputs(a,b,length,config.get('start_index',0))
            logits=a.forward(ids)
            loss=F.cross_entropy(logits[:,:-1].float().reshape(-1,logits.shape[-1]),ids[:,1:].reshape(-1)).item()
            records.append(dict(id=f'shaped/b{b}/p{length}',dataset='hardware_shape',ids=ids.cpu(),
                last_logits=logits[:,-1].float().cpu(),nll=loss,top1=logits.argmax(-1).cpu()))
    return records

def validate_references(a,ep,target,references):
    ep.count_trace=True
    domain_counts={}
    for row in references:
        ep.reset([row['id']])
        ids=row['ids'].to('cuda')
        logits=a.forward(ids)
        index=domain_counts.get(row['dataset'],0)
        domain_counts[row['dataset']]=index+1
        if ep.rank==0:
            for rec in ep.count_records:
                append(target/'routing_counts.jsonl',dict(rec,id=row['id'],dataset=row['dataset'],length=ids.shape[-1],
                    split='calibration' if row['dataset']!='hardware_shape' and index<8 else 'heldout',
                    evidence='LIVE_TRUE_EP_EXACT_ROUTER_COUNTS_FROM_ENGINEERING_CORPUS'))
        last=logits[:,-1].float()
        ref=row['last_logits'].to('cuda')
        nll=F.cross_entropy(logits[:,:-1].float().reshape(-1,logits.shape[-1]),ids[:,1:].reshape(-1)).item()
        diff=last-ref
        result=dict(id=row['id'],dataset=row['dataset'],length=ids.shape[-1],
            native_nll=row['nll'],ep_nll=nll,delta_nll=nll-row['nll'],
            last_logit_relative_l2=float(diff.norm()/ref.norm().clamp_min(1e-9)),max_logit_error=float(diff.abs().max()),
            top1_equal_count=int((logits.argmax(-1).cpu()==row['top1']).sum()),positions=ids.numel(),
            evidence='LIVE_MATCHED_NATIVE_VS_TRUE_EP_TEACHER_FORCING',
            quality_scope='Engineering fidelity. Not free-running downstream accuracy.')
        result['pass']=result['last_logit_relative_l2']<=.02 and abs(result['delta_nll'])<=.02 and result['top1_equal_count']/result['positions']>=.99
        append(target/f'engineering_quality.rank{ep.rank}.jsonl',result)
    ep.count_trace=False

def elapsed(fn):
    dist.barrier()
    torch.cuda.synchronize()
    start=time.perf_counter()
    value=fn()
    torch.cuda.synchronize()
    duration=time.perf_counter()-start
    ranks=torch.tensor([duration],device='cuda',dtype=torch.float64)
    dist.all_reduce(ranks,op=dist.ReduceOp.MAX)
    return float(ranks.item()),value

def run(a,ep,target,config):
    lengths=config.get('lengths',[128,1024,2048])
    batches=config.get('batches',[1,2,4])
    for b in batches:
        for length in lengths:
            ids,requests=shaped_inputs(a,b,length,config.get('start_index',0))
            def call():
                ep.reset(requests)
                return a.forward(ids)
            for _ in range(config.get('warmups',10)):call()
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            clean=[]
            for repeat in range(config.get('repeats',20)):
                duration,out=elapsed(call)
                clean.append(duration*1000)
                del out
            ep.trace=True
            traced_s,out=elapsed(call)
            events=ep.resolve()
            ep.trace=False
            write(target/f'trace_b{b}_p{length}.rank{ep.rank}.json',events)
            write(target/f'timing_b{b}_p{length}.rank{ep.rank}.json',dict(model=a.name,ep=ep.world,rank=ep.rank,
                batch=b,input_tokens=length,requests=requests,warmups=config.get('warmups',10),
                clean_forward_ms=clean,observer_forward_ms=traced_s*1000,
                peak_allocated_bytes=torch.cuda.max_memory_allocated(),resident_allocated_bytes=torch.cuda.memory_allocated(),
                evidence='LIVE_TRUE_EP_MODEL_ONLY_FORWARD',request_arrival='SYNCHRONIZED_EQUAL_LENGTH_COHORT',
                backend='NCCL all_gather counts + A2A dispatch + native grouped experts + A2A combine + source restoration all_reduce'))
            del out
            if config.get('micro',True):
                from .operator_controls import micro
                micro(a,ep,target,b,length)
            if ep.rank==0:print('FORWARD_CELL',a.name,ep.world,b,length,sum(clean)/len(clean),flush=True)
        if config.get('generation',True):
            ids,requests=shaped_inputs(a,b,128,config.get('start_index',0))
            for cap in config.get('generation_lengths',[32,128]):
                prefix=ids.clone()
                tokens=[]
                timestamps=[]
                dist.barrier()
                torch.cuda.synchronize()
                start=time.perf_counter()
                for step in range(cap):
                    ep.reset(requests)
                    logits=a.forward(prefix)
                    token=logits[:,-1].argmax(-1,keepdim=True)
                    torch.cuda.synchronize()
                    timestamps.append(time.perf_counter()-start)
                    tokens.append(token.cpu().flatten().tolist())
                    prefix=torch.cat([prefix,token],dim=-1)
                append(target/f'generation.rank{ep.rank}.jsonl',dict(model=a.name,ep=ep.world,rank=ep.rank,batch=b,
                    prefix=128,new_tokens=cap,requests=requests,timestamps_s=timestamps,generated_ids=tokens,
                    e2e_s=timestamps[-1],ttft_s=timestamps[0],tpot_s=(timestamps[-1]-timestamps[0])/(cap-1),
                    eos_enabled=False,evidence='LIVE_GREEDY_CAPPED_HARDWARE_COHORT_NOT_ACCURACY',
                    cache='NONE_FULL_PREFIX_RECOMPUTATION',request_arrival='SYNCHRONIZED'))
                if ep.rank==0:print('GENERATION_CELL',a.name,ep.world,b,cap,timestamps[-1],flush=True)
    if config.get('replica_memory',True):
        from .operator_controls import replica_allocation
        replica_allocation(a,ep,target)
    if config.get('profile',config.get('start_index',0)==0):
        profile_forward(a,ep,target)
    if config.get('start_index',0)>0:
        trace_generation(a,ep,target,config)

def trace_generation(a,ep,target,config):
    ids,requests=shaped_inputs(a,1,128,config.get('start_index',0))
    def generate(traced):
        prefix=ids.clone()
        output=[]
        ep.trace=traced
        dist.barrier()
        torch.cuda.synchronize()
        origin=torch.cuda.Event(enable_timing=True)
        origin.record()
        start=time.perf_counter()
        for step in range(32):
            ep.reset(requests)
            logits=a.forward(prefix)
            token=logits[:,-1].argmax(-1,keepdim=True)
            torch.cuda.synchronize()
            output.append(token.item())
            if traced:
                offset=origin.elapsed_time(ep.origin)
                for row in ep.resolve():
                    row.update(step=step,ready_generation_ms=offset+row['ready_forward_ms'],
                        complete_generation_ms=offset+row['complete_forward_ms'])
                    if ep.rank:
                        for key in ['request_expert_counts','sampled_ids','sampled_positions','requests']:
                            row.pop(key,None)
                    append(target/f'generation_trace.rank{ep.rank}.jsonl',row)
            prefix=torch.cat([prefix,token],dim=-1)
        torch.cuda.synchronize()
        wall=(time.perf_counter()-start)*1000
        ep.trace=False
        return dict(ids=output,wall_ms=wall,eos_enabled=False,cap=32,request=requests,
            evidence='LIVE_CLEAN_GENERATION_CONTROL' if not traced else 'LIVE_EVERY_STEP_CUDA_EVENTS_SEPARATE_OBSERVER')
    clean=generate(False)
    traced=generate(True)
    write(target/f'generation_trace_control.rank{ep.rank}.json',dict(clean=clean,observer=traced,
        token_exact=clean['ids']==traced['ids'],observer_ratio=traced['wall_ms']/clean['wall_ms']))
    assert clean['ids']==traced['ids'],'Generation observer changes tokens'

def profile_forward(a,ep,target):
    ids,requests=shaped_inputs(a,1,128)
    for _ in range(10):
        ep.reset(requests)
        a.forward(ids)
    dist.barrier()
    torch.cuda.synchronize()
    ep.profiler_labels=True
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,torch.profiler.ProfilerActivity.CUDA]) as profiler:
        ep.reset(requests)
        with torch.profiler.record_function('TRUE_EP_NATIVE_FULL_FORWARD'):
            a.forward(ids)
            torch.cuda.synchronize()
    ep.profiler_labels=False
    profiler.export_chrome_trace(str(target/f'native_trace.rank{ep.rank}.json.gz'))
