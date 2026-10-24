"""One independently started model process with original BF16 weights."""
import argparse
import gc
import hashlib
import json
import os
import subprocess
import time
import traceback
from pathlib import Path
import numpy as np
import torch
from .common import ROOT,FORK,MODEL,FIXTURES,UUIDS,UPSTREAM,write,digest,config

def inputs(p,n,split):
    from transformers import AutoTokenizer
    tokenizer=AutoTokenizer.from_pretrained(MODEL,local_files_only=True,trust_remote_code=True)
    manifest=json.loads((ROOT/'source_family_manifest.json').read_text())
    books=[r for r in manifest['families'] if r['split']==split]
    output=[]
    for i in range(n):
        book=books[i%len(books)]
        text=(ROOT/book['path']).read_text(encoding='utf-8-sig')
        ids=tokenizer.encode(text,add_special_tokens=False)
        offset=1024+(i//len(books))*2048
        if len(ids)<p+offset:
            others='\n\n'.join((ROOT/b['path']).read_text(encoding='utf-8-sig') for b in books if b!=book)
            ids=tokenizer.encode(text+'\n\n'+others,add_special_tokens=False)
        assert len(ids)>=p+offset,(book['family'],len(ids),p,offset)
        output.append(ids[offset:offset+p])
    return output

def generation(e,ids,cap,natural=False,profile=False,sample_memory=True):
    from vllm_rlt import SamplingParams
    params=SamplingParams(max_tokens=cap,min_loops=4,max_loops=4,exit_threshold=1.,
        temperature=0.,ignore_eos=not natural)
    outputs={};times={str(i):[] for i in range(len(ids))};memory=[]
    start=time.perf_counter()
    for i,ts in enumerate(ids):e.add_request(str(i),ts,params)
    started=False
    while e.has_unfinished_requests():
        updates=e.step()
        for out in updates:
            now=time.perf_counter();old=outputs.get(out.request_id)
            if old is None or len(out.token_ids)>len(old.token_ids):
                assert old is None or len(out.token_ids)==len(old.token_ids)+1
                times[out.request_id].append(now)
            outputs[out.request_id]=out
        if profile and not started and any(times.values()):
            e.model_runner.synchronize()
            torch.cuda.cudart().cudaProfilerStart();started=True
        if sample_memory and (len(memory)<8 or (updates and len(memory)<100)):
            cm=e.cache_manager
            memory.append(dict(host_s=time.perf_counter(),used_blocks=cm.num_used_blocks,
                allocated_bytes=torch.cuda.memory_allocated(),reserved_bytes=torch.cuda.memory_reserved(),
                allocations={rid:dict(capacity=a.max_tokens,blocks=a.block_tables,
                    initialized_prefixes=[[w.prefix for w in depth] for depth in a.written])
                    for rid,a in cm._allocations.items()}))
    e.model_runner.synchronize()
    if started:torch.cuda.cudart().cudaProfilerStop()
    data=dict(start=start,end=time.perf_counter(),memory=memory,requests=[])
    for rid,out in outputs.items():
        assert all(d==4 for d in out.exit_depths)
        data['requests'].append(dict(id=rid,tokens=list(out.token_ids),depths=list(out.exit_depths),
            times=times[rid],finish=out.finish_reason))
    return data

def layout(cm):
    """Generic layer-major storage. Preserve the original public tensor axes."""
    shape=tuple(cm.key_cache.shape);dtype=cm.key_cache.dtype
    assert not cm._allocations
    del cm.key_cache,cm.value_cache
    torch.cuda.empty_cache()
    cm.key_cache=torch.empty((shape[1],shape[0],*shape[2:]),device='cuda',dtype=dtype).transpose(0,1)
    cm.value_cache=torch.empty_like(cm.key_cache)
    # This intervention occurs before any request writes initialized KV pages.
    assert not cm._allocations
    torch.cuda.empty_cache()

def capture_fixtures(e,ids,job,target):
    """Capture Q and exact initialized K/V in the actual producing CUDA stream."""
    runner=e.model_runner;cm=e.cache_manager
    submit=runner.submit;attend=cm._attend_prepared;active=[None];seen=set();records=[]
    def issued(scheduled_batch=None,*,batch):
        active[0]=batch
        return submit(scheduled_batch,batch=batch)
    def observe(layer,batch,q):
        result=attend(layer,batch,q)
        execution=active[0]
        if layer not in (0,12,23) or execution is None or len(execution.spans)!=1:return result
        span=execution.spans[0]
        if span.token_count!=1 or span.token_start!=job['P']:return result
        depth=span.depth
        if (depth,layer) in seen:return result
        seen.add((depth,layer))
        k,v=cm.read(layer,span.request.request_id,depth,job['P']+1)
        packet=dict(q=q.detach().cpu().contiguous(),k=k.cpu().contiguous(),v=v.cpu().contiguous(),
            out=result.detach().cpu().contiguous(),P=job['P'],depth=depth,layer=layer,
            model_revision=MODEL.name,split=job.get('split','TRAIN'))
        name=f"{job.get('fixture_namespace','')}P{job['P']}_d{depth}_l{layer}.pt"
        path=FIXTURES/name
        assert not path.exists(),'Use a new immutable fixture attempt.'
        torch.save(packet,path)
        record=dict(name=name,path=str(path),sha256=digest(path),bytes=path.stat().st_size,
            depth=depth,layer=layer,sequence_length=job['P']+1,q_shape=list(q.shape),
            K_sha256=hashlib.sha256(packet['k'].view(torch.uint8).numpy().tobytes()).hexdigest(),
            V_sha256=hashlib.sha256(packet['v'].view(torch.uint8).numpy().tobytes()).hexdigest())
        records.append(record);write(target/'fixtures.json',records)
        return result
    runner.submit=issued;cm._attend_prepared=observe
    data=generation(e,ids,2)
    runner.submit=submit;cm._attend_prepared=attend
    assert len(records)==12,(len(records),sorted(seen))
    return dict(fixtures=records,data=data)

def run(model,job,target):
    if job['kind']=='local_split':
        from .local_split import run as segmented_control
        return segmented_control(job,target)
    if job['kind']=='tiering':
        from .tiering import run as transfer_test
        return transfer_test(job,target)
    from vllm_rlt.engine.stage_engine import StageEngine
    p=job.get('P',2048);n=job.get('N',1);cap=job.get('cap',64)
    ids=inputs(p,n,job.get('split','TRAIN'))
    write(target/'inputs.json',dict(P=p,N=n,token_ids=ids,split=job.get('split','TRAIN')))
    blocks=job.get('blocks',18432)
    required=n*((p+cap+15)//16)*4
    assert required<=blocks,('CAPACITY_REJECTED_BEFORE_ALLOCATION',required,blocks)
    assert p+cap<=65536
    graphs=job.get('graphs',True)
    e=StageEngine(model,**config(n,blocks,graphs))
    if job.get('layout')=='layer_major':layout(e.cache_manager)
    cm=e.cache_manager
    facts=dict(cache_shape=list(cm.key_cache.shape),cache_stride=list(cm.key_cache.stride()),
        pool_bytes=cm.key_cache.numel()*cm.key_cache.element_size()*2,
        element_bytes=cm.key_cache.element_size(),cache_storage_depths=cm.storage_depths,
        backend=cm.attention_info,logical_prompt_payload_bytes=p*n*786432,
        reserved_output_capacity_blocks=required,num_blocks=blocks,
        allocated_after_engine=torch.cuda.memory_allocated(),reserved_after_engine=torch.cuda.memory_reserved(),
        layout=job.get('layout','native_block_major'))
    write(target/'memory_facts.json',facts)
    if job['kind']=='capacity':
        admissions=[]
        for length in (1024,8192,16384,32768,65408):
            accepted=[]
            while True:
                rid=f'admission_{length}_{len(accepted)}'
                if not cm.allocate(rid,length+64):break
                accepted.append(rid)
            admissions.append(dict(P=length,admitted_empty_reservations=len(accepted),
                first_rejection_index=len(accepted),free_blocks=cm.num_free_blocks,
                reserved_blocks=cm.num_used_blocks,evidence='LIVE_MEASURED_OPERATOR',
                qualifier='Actual allocator admission, uninitialized pages. Not completed concurrent inference capacity.'))
            for rid in accepted:cm.free(rid)
        write(target/'capacity_admission.json',admissions)
        result=dict(rows=admissions)
    elif job['kind']=='capture':
        result=capture_fixtures(e,ids,job,target)
    else:
        generation(e,ids,8)
        if job.get('barrier'):
            write(target/'READY.json',dict(wall_s=time.time()))
            path=ROOT/'state'/f"release_{job['barrier']}.json"
            until=time.monotonic()+1800
            while not path.exists():
                assert time.monotonic()<until
                time.sleep(.2)
            start=json.loads(path.read_text())['start_wall_s']
            while time.time()<start:time.sleep(.01)
        rows=[]
        for repeat in range(job.get('repeats',3)):
            torch.cuda.reset_peak_memory_stats()
            data=generation(e,ids,cap,natural=job.get('natural',False),profile=job.get('nsys',False),sample_memory=job.get('sample_memory',True))
            gaps=[(b-a)*1000 for r in data['requests'] for a,b in zip(r['times'],r['times'][1:])]
            ttft=[(r['times'][0]-data['start'])*1000 for r in data['requests']]
            first=max(r['times'][0] for r in data['requests']);last=max(r['times'][-1] for r in data['requests'])
            item=dict(P=p,N=n,repeat=repeat,gpu=int(os.environ['CUDA_VISIBLE_DEVICES']),
                evidence='LIVE_MEASURED_SERVING',lane='HEAVY_PROFILE' if job.get('nsys') else 'CLEAN_BOUNDED_HARDWARE',
                natural=job.get('natural',False),TTFT_mean_ms=float(np.mean(ttft)),
                TTFT_P95_ms=float(np.percentile(ttft,95)),E2E_cohort_ms=(data['end']-data['start'])*1000,
                ITL_mean_ms=float(np.mean(gaps)),ITL_P50_ms=float(np.percentile(gaps,50)),
                ITL_P95_ms=float(np.percentile(gaps,95)),ITL_P99_ms=float(np.percentile(gaps,99)),
                decode_span_ms=(last-first)*1000,decode_tokens_s=sum(sum(t>first for t in r['times']) for r in data['requests'])/(last-first),
                peak_allocated=torch.cuda.max_memory_allocated(),peak_reserved=torch.cuda.max_memory_reserved(),
                graph_captures=e.model_runner.graphs.captures if e.model_runner.graphs else 0,
                graph_replays=e.model_runner.graphs.replays if e.model_runner.graphs else 0,
                data=data)
            write(target/f'measured_{repeat}.json',item)
            rows.append({k:v for k,v in item.items() if k!='data'})
        result=dict(rows=rows)
    e.close();del cm,e;gc.collect();torch.cuda.empty_cache()
    return result

@torch.inference_mode()
def main():
    parser=argparse.ArgumentParser();parser.add_argument('--job',required=True);args=parser.parse_args()
    job=json.loads(Path(args.job).read_text());target=ROOT/'raw'/job['id'];target.mkdir(parents=True,exist_ok=True)
    assert not (target/'completed.json').exists()
    gpu=int(os.environ['CUDA_VISIBLE_DEVICES']);assert gpu in UUIDS and torch.cuda.device_count()==1
    actual=str(torch.cuda.get_device_properties(0).uuid)
    assert actual.removeprefix('GPU-')==UUIDS[gpu].removeprefix('GPU-')
    torch.set_num_threads(1)
    import vllm_rlt
    assert Path(vllm_rlt.__file__).resolve().parent==FORK/'vllm_rlt'
    assert not subprocess.check_output(['git','diff','--binary'],cwd=FORK)
    write(target/'identity.json',dict(job=job,physical_gpu=gpu,uuid=actual,pid=os.getpid(),uid=os.getuid(),
        source=UPSTREAM,model=MODEL.name,wall_start=time.time(),affinity=sorted(os.sched_getaffinity(0))))
    write(target/'source_snapshot.json',{p.name:dict(sha256=digest(p),text=p.read_text()) for p in Path(__file__).parent.glob('*.py')})
    try:
        from vllm_rlt.models import OuroForCausalLM
        model=OuroForCausalLM.from_pretrained(MODEL,device='cuda:0',dtype=torch.bfloat16)
        assert len(model.model.layers)==24 and model.config.total_ut_steps==4
        assert all(p.dtype==torch.bfloat16 for p in model.parameters())
        storages={p.untyped_storage().data_ptr():p.untyped_storage().nbytes() for p in model.parameters()}
        write(target/'model_facts.json',dict(parameters=sum(p.numel() for p in model.parameters()),
            weight_bytes=sum(storages.values()),config=model.config.to_dict()))
        result=run(model,job,target)
        artifacts=[dict(path=p.name,bytes=p.stat().st_size,sha256=digest(p)) for p in target.iterdir()
            if p.is_file() and p.name not in ('completed.json','telemetry.jsonl','READY.json')]
        write(target/'completed.json',dict(summary=result,artifacts=artifacts,end_wall_s=time.time()))
    except BaseException as error:
        write(target/'failure.json',dict(error=repr(error),traceback=traceback.format_exc(),wall_s=time.time()))
        raise

if __name__=='__main__':main()
