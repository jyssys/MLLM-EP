"""Aggregate completed real EP runs. Observer spans never form a clean 100% pie."""
from collections import defaultdict
import csv
import json
from pathlib import Path
import numpy as np
from .common import *

def rows(path):
    return [json.loads(line) for line in path.open()]

def table(path,values):
    path=ROOT/path
    path.parent.mkdir(parents=True,exist_ok=True)
    keys=list(dict.fromkeys(k for r in values for k in r))
    with path.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=keys,lineterminator='\n')
        w.writeheader();w.writerows(values)

def ci(values):
    x=np.array(values,float)
    rng=np.random.default_rng(SEED)
    boot=x[rng.integers(len(x),size=(4000,len(x)))].mean(1)
    return dict(mean=float(x.mean()),p50=float(np.median(x)),p90=float(np.quantile(x,.9)),p95=float(np.quantile(x,.95)),
        ci95_lo=float(np.quantile(boot,.025)),ci95_hi=float(np.quantile(boot,.975)),n=len(x))

def completed():
    return [p for p in sorted((ROOT/'raw').iterdir()) if (p/'complete.json').exists()]

def main():
    timing=defaultdict(list)
    fidelity=[];quality=[];gen=[];imbalance=[];components=[];overlap=[];memory=[];lookahead=[]
    eos={}
    for p in (ROOT/'raw').glob('*/native_metadata.rank0.json'):
        r=json.loads(p.read_text())
        if 'eos_token_id' in r:eos[r['model']]=r['eos_token_id']
    for run in completed():
        config=json.loads(next(run.glob('request.rank*.json')).read_text())['config']
        model=config['model'];world=len(config['gpus'])
        f=run/'fidelity.rank0.json'
        if f.exists():
            q=json.loads(f.read_text())
            fidelity.append(dict(model=model,ep=world,run=run.name,physical_gpus=str(config['gpus']),
                relative_l2=q['relative_l2'],max_logit_error=q['max_absolute_error'],delta_nll=q['delta_nll'],
                token_exact=q['token_exact'],pass_fidelity=q['pass_fidelity'],
                routed_bytes_rank0=q['routed_weight_bytes'],resident_after_shard_rank0=q['resident_after_sharding']))
        f=run/'engineering_quality.rank0.jsonl'
        if f.exists():quality.extend(dict(r,model=model,ep=world,run=run.name) for r in rows(f))
        for f in run.glob('timing_b*_p*.rank0.json'):
            t=json.loads(f.read_text());timing[(model,world,t['batch'],t['input_tokens'])].append((run.name,t))
            traces=[json.loads((run/f'trace_b{t["batch"]}_p{t["input_tokens"]}.rank{rank}.json').read_text()) for rank in range(world)]
            # One cumulative critical rank, not independent maxima of each category.
            critical=max(range(world),key=lambda rank:sum(r['total_span_ms'] for r in traces[rank]))
            names=['metadata_pack','dispatch','receive_grouping','expert','combine','restore']
            component=dict(model=model,ep=world,run=run.name,batch=t['batch'],prefix=t['input_tokens'],
                selected_cumulative_rank=critical,clean_mean_ms=np.mean(t['clean_forward_ms']),observer_ms=t['observer_forward_ms'],
                observer_ratio=t['observer_forward_ms']/np.mean(t['clean_forward_ms']),evidence='SEPARATE_OBSERVER_API_SPANS_PARTIAL_CRITICAL_PATH')
            component.update({name+'_ms':sum(r[name+'_ms'] for r in traces[critical]) for name in names})
            components.append(component)
            for call in traces[0]:
                load=np.array(call['rank_loads'],float)
                imbalance.append(dict(model=model,ep=world,run=run.name,batch=t['batch'],prefix=t['input_tokens'],
                    site=call['site'],loop=call['loop'],recurrent=call['recurrent'],
                    rank_counts=json.dumps(call['rank_loads']),cv=float(load.std()/max(load.mean(),1e-9)),
                    max_mean=float(load.max()/max(load.mean(),1e-9)),expert_counts=json.dumps(call['expert_counts']),
                    evidence='LIVE_TRUE_EP_EXACT_ROUTER_COUNTS'))
            history={}
            for call in traces[0]:
                if not call['recurrent']:continue
                last=history.get(call['site'])
                if last:
                    a=np.array(call['sampled_ids']);b=np.array(last['sampled_ids'])
                    if a.shape==b.shape:
                        intersection=np.array([len(set(x)&set(y)) for x,y in zip(a,b)])
                        lookahead.append(dict(model=model,ep=world,run=run.name,batch=t['batch'],prefix=t['input_tokens'],
                            site=call['site'],loop=call['loop'],positions=len(a),
                            recall=float(np.mean(intersection/2)),precision=float(np.mean(intersection/2)),
                            jaccard=float(np.mean(intersection/(4-intersection))),
                            evidence='LIVE_ROUTE_LOOKAHEAD_DIAGNOSTIC_NO_PREDISPATCH'))
                history[call['site']]=call
        f=run/'generation.rank0.jsonl'
        if f.exists():
            for r in rows(f):
                its=np.diff(r['timestamps_s'])*1000
                gen.append(dict(model=model,ep=world,run=run.name,batch=r['batch'],prefix=r['prefix'],
                    new_tokens=r['new_tokens'],e2e_ms=r['e2e_s']*1000,ttft_ms=r['ttft_s']*1000,
                    tpot_ms=r['tpot_s']*1000,itl_p95_ms=np.quantile(its,.95),
                    output_tokens_per_s=r['batch']*r['new_tokens']/r['e2e_s'],
                    requests_per_s=r['batch']/r['e2e_s'],eos_enabled=False,
                    eos_tokens_observed=sum(token==eos.get(model) for step in r['generated_ids'] for token in step),
                    stop_reason='HARDWARE_CAP_EOS_DISABLED',shape_warmup='FIRST_OCCURRENCE_SHAPES_NOT_FULLY_WARMED',
                    evidence='LIVE_SYNCHRONIZED_HARDWARE_GENERATION_NOT_NATURAL_ACCURACY'))
        for f in run.glob('overlap.rank*.jsonl'):
            for r in rows(f):
                overlap.append(dict(model=model,ep=world,run=run.name,rank=r['rank'],
                    sequential_ms=np.mean(r['sequential_ms']),overlap_ms=np.mean(r['overlap_ms']),
                    operator_ratio=np.mean(r['sequential_ms'])/np.mean(r['overlap_ms']),
                    evidence='LIVE_OPERATOR_CONTROL_NOT_REQUEST_E2E'))
        for f in run.glob('replica_memory.rank*.jsonl'):
            for r in rows(f):
                memory.append({k:v for k,v in dict(r,run=run.name).items() if k!='copies'})
    timing_rows=[]
    for (model,ep,b,p),values in sorted(timing.items()):
        starts=[np.mean(t['clean_forward_ms']) for _,t in values]
        ci_values=starts if len(starts)>1 else values[0][1]['clean_forward_ms']
        timing_rows.append(dict(model=model,ep=ep,batch=b,prefix=p,starts=len(values),**ci(ci_values),
            ci_unit='PROCESS_START_AND_PROMPT_COHORT_MEAN' if len(starts)>1 else 'SERIAL_REPETITION_EXPLORATORY',
            mean_ms=float(np.mean(starts)),peak_allocated_bytes=max(t['peak_allocated_bytes'] for _,t in values),
            source_restoration='INCLUDED',evidence='LIVE_TRUE_EP_COMPLETE_FORWARD'))
    for name,values in [('baseline_forward',timing_rows),('native_fidelity',fidelity),('engineering_quality',quality),
            ('generation',gen),('rank_loads',imbalance),('components',components),('overlap_operator',overlap),
            ('replica_memory',memory),('route_lookahead',lookahead)]:
        table('summary/'+name+'.csv',values)
    write(ROOT/'summary/status.json',dict(completed_runs=len(completed()),baseline_cells=len(timing_rows),
        engineering_examples=len(quality),timestamp=__import__('time').time()))
    print('SUMMARY',len(timing_rows),len(quality),len(gen))

if __name__=='__main__':main()
