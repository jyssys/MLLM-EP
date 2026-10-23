"""Pretrained correctness and completed-output-token timing experiments."""
import gc
import hashlib
import json
import time
from dataclasses import replace
import numpy as np
import torch
from .common import ROOT,MODEL,write,config,guard

def new_engine(model,N,D,job):
    from vllm_rlt.engine.stage_engine import StageEngine
    return StageEngine(model,**config(N,D,graphs=job.get('graphs',True),
        asynchronous=job.get('async',True),mode=job.get('mode','refill'),blocks=job.get('blocks',18432)))

def inputs(P,N):
    from transformers import AutoTokenizer
    tokenizer=AutoTokenizer.from_pretrained(MODEL,local_files_only=True,trust_remote_code=True)
    base=tokenizer.encode('A farmer measures a rectangle with length twelve and width seven. Explain the calculation carefully. '*1800)
    return [base[i:i+P] for i in range(N)]

def generate(e,ids,cap=8,ignore_eos=True,decode_barrier=False):
    from vllm_rlt import SamplingParams
    params=SamplingParams(max_tokens=cap,min_loops=4,max_loops=4,exit_threshold=1.,temperature=0.,ignore_eos=ignore_eos)
    for i,ts in enumerate(ids):e.add_request(str(i),ts,params)
    original_ready=e.scheduler.execution_ready
    released=[False]
    def ready(request):
        if not released[0] and len(e.scheduler.requests)==len(ids) and all(r.generated_token_ids for r in e.scheduler.requests.values()):
            released[0]=True
        return (released[0] or request.num_scheduled_outputs==0) and (original_ready is None or original_ready(request))
    if decode_barrier:e.scheduler.execution_ready=ready
    times={str(i):[] for i in range(len(ids))};outputs={};steps=0
    start=time.perf_counter()
    while e.has_unfinished_requests():
        for out in e.step():
            now=time.perf_counter();old=outputs.get(out.request_id)
            if old is None or len(out.token_ids)>len(old.token_ids):
                assert old is None or len(out.token_ids)==len(old.token_ids)+1
                times[out.request_id].append(now)
            outputs[out.request_id]=out
        steps+=1
        assert steps<200000
    e.model_runner.synchronize()
    e.scheduler.execution_ready=original_ready
    assert all(len(o.token_ids)==len(times[r]) and all(d==4 for d in o.exit_depths) for r,o in outputs.items())
    return dict(start=start,end=time.perf_counter(),steps=steps,requests=[dict(id=r,tokens=list(o.token_ids),
        depths=list(o.exit_depths),times=times[r],finish=o.finish_reason) for r,o in outputs.items()])

def numerical(model,job,target):
    """Heavy capture occurs inside the producer stream. It is not clean timing."""
    from vllm_rlt.request import Operation
    rows=[]
    for P,N in job['shapes']:
        ids=inputs(P,N);reference=None
        for D in job.get('groups',[1,2,4,3]):
            e=new_engine(model,N,D,job);runner=e.model_runner;original=runner.runtime.forward
            snapshots=[];leases=[];submit=runner.submit;active=[None]
            def traced_submit(scheduled_batch=None,*,batch):
                active[0]=batch
                if batch.operation==Operation.RECURRENT:
                    assert all(s.depth+batch.loop_count<=4 for s in batch.spans)
                    leases.append(dict(k=batch.loop_count,ids=[s.request.request_id for s in batch.spans],
                        depths=[s.depth for s in batch.spans],positions=[s.token_start for s in batch.spans],
                        token_counts=[s.token_count for s in batch.spans],chunked=batch.has_chunks))
                return submit(scheduled_batch,batch=batch)
            def capture(batch):
                result=original(batch)
                if batch.operation==Operation.CODA:
                    # The runtime producer is on this CUDA stream. Blocking CPU copies
                    # wait for it before diagnostic reads. No unfinished state is read.
                    for row,span in enumerate(active[0].spans):
                        index=span.request.num_scheduled_outputs
                        if index>=3:continue
                        values=[]
                        for depth in range(4):
                            for layer in range(24):
                                k,v=e.cache_manager.read(layer,span.request.request_id,depth,span.token_start+1)
                                positions=sorted(set((0,max(0,span.token_start-1),span.token_start)))
                                values.extend((k[positions].clone().cpu(),v[positions].clone().cpu()))
                        snapshots.append(dict(id=span.request.request_id,index=index,position=span.token_start,
                            hidden=batch.hidden[row].clone().cpu(),logits=result.logits[row].clone().cpu(),KV=values))
                return result
            runner.runtime.forward=capture;runner.submit=traced_submit
            data=generate(e,ids,cap=4,decode_barrier=job.get('decode_barrier',False))
            snapshots.sort(key=lambda s:(s['id'],s['index']))
            snapshot_hash=[dict(id=s['id'],index=s['index'],position=s['position'],hidden=hashlib.sha256(s['hidden'].view(torch.uint8).numpy().tobytes()).hexdigest(),
                logits=hashlib.sha256(s['logits'].view(torch.uint8).numpy().tobytes()).hexdigest(),
                KV_sha256=hashlib.sha256(b''.join(v.view(torch.uint8).numpy().tobytes() for v in s['KV'])).hexdigest(),
                KV_by_depth_sha256=[hashlib.sha256(b''.join(v.view(torch.uint8).numpy().tobytes() for v in s['KV'][d*48:(d+1)*48])).hexdigest() for d in range(4)],
                hidden_norm=float(torch.linalg.vector_norm(s['hidden'].float())),
                KV_tensors=len(s['KV']),top_two=torch.topk(s['logits'].float(),2,dim=-1).values.tolist()) for s in snapshots]
            item=dict(P=P,N=N,D=D,graphs=job.get('graphs',True),asynchronous=job.get('async',True),
                input_ids=ids,tokens=[r['tokens'] for r in sorted(data['requests'],key=lambda r:r['id'])],
                snapshots=snapshot_hash,leases=leases,depths=[r['depths'] for r in data['requests']],lane='HEAVY_CORRECTNESS')
            item['prompt_token_loop_work']=sum(x['k']*sum(x['token_counts']) for x in leases if x['chunked'])
            assert item['prompt_token_loop_work']==N*P*4
            if reference is None:reference=(item['tokens'],snapshots)
            else:
                tokens,ref=reference
                shape_match=len(ref)==len(snapshots) and all((a['id'],a['index'],a['position'],a['hidden'].shape)==(b['id'],b['index'],b['position'],b['hidden'].shape) for a,b in zip(ref,snapshots))
                metrics=[]
                if shape_match:
                    for a,b in zip(ref,snapshots):
                        for key in ('hidden','logits'):
                            av,bv=a[key].float(),b[key].float()
                            metrics.append(float(torch.linalg.vector_norm(av-bv)/torch.linalg.vector_norm(av).clamp_min(1e-12)))
                        assert len(a['KV'])==len(b['KV'])
                        for av,bv in zip(a['KV'],b['KV']):
                            metrics.append(float(torch.linalg.vector_norm(av.float()-bv.float())/torch.linalg.vector_norm(av.float()).clamp_min(1e-12)))
                item['comparison']=dict(tokens_exact=tokens==item['tokens'],snapshot_shapes_match=shape_match,
                    KV_exact=shape_match and all(torch.equal(av,bv) for a,b in zip(ref,snapshots) for av,bv in zip(a['KV'],b['KV'])),
                    max_relative_L2=max(metrics) if metrics else None,tolerance=.03,
                    numerical_pass=bool(metrics) and max(metrics)<=.03)
                item['comparison']['first_nonexact_snapshot']=next((dict(id=a['id'],output_index=a['index'],position=a['position']) for a,b in zip(ref,snapshots)
                    if not torch.equal(a['hidden'],b['hidden']) or not torch.equal(a['logits'],b['logits']) or any(not torch.equal(av,bv) for av,bv in zip(a['KV'],b['KV']))),None)
            write(target/f'numeric_P{P}_N{N}_D{D}.json',item)
            rows.append({k:v for k,v in item.items() if k not in ('leases','snapshots','tokens')})
            runner.runtime.forward=original;runner.submit=submit
            del original,submit,capture,traced_submit,snapshots,runner
            e.close();del e;gc.collect();torch.cuda.empty_cache()
        del reference
    assert all(r.get('comparison',{}).get('numerical_pass',True) for r in rows),rows
    return dict(rows=rows)

def timing(model,job,target):
    rows=[]
    for P,N in job['shapes']:
        ids=inputs(P,N)
        for D in job.get('groups',[1,2,4]):
            e=new_engine(model,N,D,job)
            fairness=None
            if job.get('fairness')=='completion':
                from ..ouro_budget_joint_cg.fairness import attach
                fairness=attach(e,True)
            generate(e,ids,cap=8)
            if job.get('barrier'):
                write(target/'READY.json',dict(time=time.time()))
                release=ROOT/'state'/f"release_{job['barrier']}.json"
                until=time.monotonic()+1800
                while not release.exists():
                    assert time.monotonic()<until
                    time.sleep(.2)
                scheduled=json.loads(release.read_text())['start_wall_s']
                while time.time()<scheduled:time.sleep(.01)
            for repeat in range(job.get('repeats',3)):
                torch.cuda.reset_peak_memory_stats();before=time.time()
                data=generate(e,ids,cap=job.get('cap',64),ignore_eos=job.get('ignore_eos',True),decode_barrier=job.get('decode_barrier',False))
                gaps=[b-a for r in data['requests'] for a,b in zip(r['times'],r['times'][1:])]
                # Prefill/first output is excluded. Host-delivered completed tokens
                # supply the decode measurement, not individual recurrent callbacks.
                first=max(r['times'][0] for r in data['requests'])
                last=max(r['times'][-1] for r in data['requests'])
                completed=sum(sum(t>first for t in r['times']) for r in data['requests'])
                graph=e.model_runner.graphs
                item=dict(P=P,N=N,D=D,repeat=repeat,wall_start=before,wall_end=time.time(),
                    gpu=guard(),graphs=job.get('graphs',True),asynchronous=job.get('async',True),
                    runtime_mode=job.get('mode','refill'),
                    fairness=job.get('fairness','stock'),
                    decode_barrier=job.get('decode_barrier',False),
                    decode_span_s=last-first,completed_tokens=completed,decode_tokens_s=completed/(last-first),
                    ITL_mean_ms=float(np.mean(gaps)*1000),ITL_P50_ms=float(np.percentile(gaps,50)*1000),
                    ITL_P95_ms=float(np.percentile(gaps,95)*1000),ITL_P99_ms=float(np.percentile(gaps,99)*1000),
                    peak_bytes=torch.cuda.max_memory_allocated(),graph_captures=graph.captures if graph else 0,
                    graph_replays=graph.replays if graph else 0,graph_fallbacks=graph.fallbacks if graph else 0,
                    lane=job.get('lane','PARALLEL_CHARACTERIZATION'),evidence='LIVE_MEASURED',data=data)
                write(target/f'timing_P{P}_N{N}_D{D}_r{repeat}.json',item);rows.append({k:v for k,v in item.items() if k!='data'})
            del graph
            e.close();del e,fairness;gc.collect();torch.cuda.empty_cache()
    return dict(rows=rows)

def run_job(model,job,target):
    if job['kind']=='natural_stops':
        from .natural_stops import run_job as stops
        return stops(model,job,target)
    if job['kind']=='nsight':
        from .nsight_probe import run_job as nsight
        return nsight(model,job,target)
    if job['kind']=='lifecycle':
        from .lifecycle import run_job as lifecycle
        return lifecycle(model,job,target)
    if job['kind']=='profile':
        from .profile import run_job as profile
        return profile(model,job,target)
    if job['kind']=='intervention':
        from .interventions import run_job as intervene
        return intervene(model,job,target)
    return {'numerical':numerical,'timing':timing}[job['kind']](model,job,target)
