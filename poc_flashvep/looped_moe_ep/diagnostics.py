"""Analyze actual ready snapshots and profiler intervals. No speedup simulation."""
from collections import defaultdict
import gzip
import json
import numpy as np
from .common import *
from .analyze import rows,table,completed

def union(intervals):
    result=[]
    for start,end in sorted(intervals):
        if result and start<=result[-1][1]:result[-1]=(result[-1][0],max(end,result[-1][1]))
        else:result.append((start,end))
    return result

def intersect(a,b):
    i=j=0;total=0.
    while i<len(a) and j<len(b):
        total+=max(0,min(a[i][1],b[j][1])-max(a[i][0],b[j][0]))
        if a[i][1]<=b[j][1]:i+=1
        else:j+=1
    return total

def key(work):
    return (work['request'],work['site'],work['loop'],work['step'])

def main():
    ready_summary=[];kernel_overlap=[];steady=[];drift=[]
    for run in completed():
        config=json.loads(next(run.glob('request.rank*.json')).read_text())['config']
        model=config['model'];world=len(config['gpus'])
        f=run/'ready_queue.jsonl'
        if f.exists():
            groups=defaultdict(list)
            for r in rows(f):groups[(r['batch'],r['arrival'])].append(r)
            cases=json.loads((run/'ready_probe.rank0.json').read_text())['cases']
            for (batch,arrival),events in groups.items():
                chosen_service={}
                eligible=set();same=set();total_work=0;eligible_work=0;waits=[]
                for event in events:
                    req,site_idx,loop,step=event['chosen']
                    picked=next(w for w in event.get('rank0_ready',event['ready']) if (w['request'],w['loop'],w['step'])==(req,loop,step))
                    chosen_service[key(picked)]=event['ep_service_span_ms']
                    total_work+=sum(picked['expert_counts'])
                    waits.append(event['chosen_host_s']-picked['ready_host_s'])
                    for a in event['ready']:
                        for b in event['ready']:
                            if a['request']==b['request'] or a['site']!=b['site']:continue
                            overlap=sum(min(x,y) for x,y in zip(a['expert_counts'],b['expert_counts']))
                            if not overlap:continue
                            if a['loop']!=b['loop']:eligible.add(key(a));eligible.add(key(b))
                            else:same.add(key(a));same.add(key(b))
                case=next(c for c in cases if c['batch']==batch and c['arrival']==arrival)
                saving=sum(chosen_service.get(k,0) for k in eligible)
                wall=case['diagnostic_wall_s']*1000
                ready_summary.append(dict(model=model,ep=world,batch=batch,arrival=arrival,decisions=len(events),
                    distinct_cross_pass_eligible_calls=len(eligible),same_pass_eligible_calls=len(same),
                    eligible_call_fraction=len(eligible)/max(len(events),1),
                    cross_eligible_ep_span_ms=saving,diagnostic_wall_ms=wall,
                    erase_all_eligible_calls_ceiling=wall/max(wall-saving,wall*.01),
                    mean_observed_queue_wait_ms=1000*np.mean(waits),p95_observed_queue_wait_ms=1000*np.quantile(waits,.95),
                    token_exact=case['token_exact'],
                    evidence='LIVE_INDEPENDENT_READY_REQUEST_SNAPSHOTS_NO_COALESCING',
                    ceiling_scope='Diagnostic observer only. Erases complete eligible API spans, including impossible savings. Not a production E2E bound.'))
        for f in run.glob('overlap_trace.rank*.json.gz'):
            rank=int(f.name.split('.rank')[1].split('.')[0])
            with gzip.open(f,'rt') as stream:trace=json.load(stream)
            kernels=[r for r in trace['traceEvents'] if r.get('cat')=='kernel' and 'dur' in r]
            comm=union([(r['ts'],r['ts']+r['dur']) for r in kernels if 'nccl' in r.get('name','').lower()])
            compute=union([(r['ts'],r['ts']+r['dur']) for r in kernels if 'nccl' not in r.get('name','').lower()])
            comm_us=sum(e-s for s,e in comm);compute_us=sum(e-s for s,e in compute)
            kernel_overlap.append(dict(model=model,ep=world,run=run.name,rank=rank,nccl_kernel_union_us=comm_us,
                compute_kernel_union_us=compute_us,available_overlap_us=min(comm_us,compute_us),
                achieved_overlap_us=intersect(comm,compute),
                request_critical_path_saved_us=None,evidence='HEAVY_OPERATOR_TIMELINE_NOT_CLEAN_E2E'))
        f=run/'steady_generation.rank0.jsonl'
        if f.exists():
            for r in rows(f):
                steady.append(dict(model=model,ep=world,run=run.name,trial=r['trial'],kind=r['kind'],cap=r['cap'],
                    e2e_ms=r['e2e_s']*1000,ttft_ms=r['ttft_s']*1000,tpot_ms=r['tpot_s']*1000,
                    evidence='LIVE_UNINSTRUMENTED_B1_GENERATION',warmup_generations=r['warmup_full_generations']))
            drift.append(dict(model=model,ep=world,run=run.name,**json.loads((run/'shape_drift.rank0.json').read_text())))
    table('summary/ready_opportunity.csv',ready_summary)
    table('summary/kernel_overlap.csv',kernel_overlap)
    table('summary/steady_generation.csv',steady)
    table('summary/shape_drift.csv',drift)
    print('DIAGNOSTICS',len(ready_summary),len(kernel_overlap),len(steady))

if __name__=='__main__':main()
