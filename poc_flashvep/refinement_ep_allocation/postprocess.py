"""Score completed requests without interpreting partial cohorts as full evaluations."""
import argparse
import csv
import fcntl
import hashlib
import json
import statistics
import sys
from pathlib import Path
import numpy as np
from .common import ROOT, REFERENCE, append, write, rows
from .audit import wilson

def percentile(values,q):
    return float(np.percentile(values,q)) if values else None

def paired_bootstrap(first, second, seed=20261027, repetitions=10000):
    if not first:
        return None
    delta=np.array(second,dtype=float)-np.array(first,dtype=float)
    rng=np.random.default_rng(seed)
    means=[]
    for _ in range(repetitions):
        means.append(float(rng.choice(delta,size=len(delta),replace=True).mean()))
    return [100*percentile(means,2.5),100*percentile(means,97.5)]

def normalize_native_record(record):
    """Keep raw records intact. Apply source-derived logging corrections once."""
    r=dict(record)
    if r.get('nfe_evidence') not in ('MEASURED_MODEL_FORWARD_COUNTER','SOURCE_DERIVED_PREFILL_COUNTER_CORRECTION'):
        prefill=max(r['prompt_tokens']//32*32,32)
        prefill_forwards=1+max(0,(prefill-128)//32)
        r['original_logged_nfe']=r['nfe']
        r['nfe']=r['nfe']+prefill_forwards-1
        r['nfe_evidence']='SOURCE_DERIVED_PREFILL_COUNTER_CORRECTION'
        r['num_explicit_prefill_forwards']=prefill_forwards
    if r.get('termination_evidence') not in ('MEASURED_TERMINAL_BUFFER','SOURCE_DERIVED_FROM_MEASURED_RETURN_LENGTH'):
        r['original_logged_termination_reason']=r['termination_reason']
        capacity=((r['prompt_tokens']+r['gen_length_cap']+31)//32)*32-r['prompt_tokens']
        removed=capacity-r['generation_tokens']
        assert removed>=0
        r['termination_reason']='EOS_FILTERED_SOURCE_DERIVED' if removed else r['termination_reason']
        r['termination_evidence']='SOURCE_DERIVED_FROM_MEASURED_RETURN_LENGTH'
        r['native_effective_generation_capacity']=capacity
        r['native_filtered_eos_count']=removed
    return r

def collect(include_hooked=False):
    cohorts={}
    for path in (ROOT/'hf').glob('*/shard*.jsonl'):
        records=rows(path,allow_incomplete=True)
        for r in records:
            if r['task'] in ('gsm8k','humaneval') and r.get('split','').endswith('_test'):
                key=('REFERENCE_HF',r['task'],r['run_id'])
                cohorts.setdefault(key,[]).append(r)
    for path in (ROOT/'ep4').glob('*/quality_samples.jsonl'):
        for r in rows(path,allow_incomplete=True):
            if r['task'] in ('gsm8k','humaneval') and r.get('split','').endswith('_test') and (r['mode']=='bypass' or (include_hooked and r['mode']=='hooked_noop')):
                r=normalize_native_record(r)
                key=('P0_HOOKED_EP4' if r['mode']=='hooked_noop' else 'P0_FULL_EP4',r['task'],r['run_id'])
                cohorts.setdefault(key,[]).append(r)
    return cohorts

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--score-humaneval',action='store_true')
    p.add_argument('--pilot',action='store_true')
    a=p.parse_args()
    scoring_lock=(ROOT/'state/scoring.lock').open('a')
    fcntl.flock(scoring_lock,fcntl.LOCK_EX)
    cohorts=collect(include_hooked=a.pilot)
    destination=ROOT/'pilot' if a.pilot else ROOT
    destination.mkdir(exist_ok=True)
    if a.pilot:
        for method in ('REFERENCE_HF','P0_FULL_EP4'):
            saved=ROOT/'pilot'/f'gsm8k_reused_{method}.jsonl'
            if saved.exists():
                data=rows(saved)
                if method=='P0_FULL_EP4':
                    data=[normalize_native_record(r) for r in data]
                cohorts[(method,'gsm8k',data[0]['run_id'])]=data
        allowed={task:{r['sample_id'] for r in rows(ROOT/'pilot'/f'{task}_manifest.jsonl')}
                 for task in ('gsm8k','humaneval')}
        cohorts={key:[r for r in data if r['sample_id'] in allowed[key[1]]] for key,data in cohorts.items()}
        cohorts={key:data for key,data in cohorts.items() if data}
    sys.path.insert(0,str(REFERENCE))
    from score_humaneval import sandbox_check
    cachepath=ROOT/'human_eval_scoring_cache.json'
    cache=json.loads(cachepath.read_text()) if cachepath.exists() else {}
    summaries=[]
    samples=[]
    for (method,task,runid),data in sorted(cohorts.items()):
        unique={r['sample_id']:r for r in data}
        assert len(unique)==len(data), f'Duplicate sample IDs in {runid}'
        data=sorted(data,key=lambda r:r['id'])
        for r in data:
            if task=='humaneval':
                key=hashlib.sha256(json.dumps({k:r[k] for k in ('postprocessed_generation',
                    'original_prompt','test','entry_point')},sort_keys=True).encode()).hexdigest()
                if key not in cache and a.score_humaneval:
                    cache[key]=sandbox_check(r)
                    write(cachepath,cache)
                r['humaneval_result']=cache.get(key)
                r['correct']=cache[key]['passed'] if key in cache else None
            samples.append(dict(r,scoring_method=method))
        evaluated=[r for r in data if r['correct'] is not None]
        n=len(evaluated)
        k=sum(bool(r['correct']) for r in evaluated)
        official=94.24 if task=='gsm8k' else 86.59
        expected=(128 if task=='gsm8k' else 32) if a.pilot else (1319 if task=='gsm8k' else 164)
        summary=dict(method=method,task=task,run_id=runid,generated_n=len(data),scored_n=n,
            expected_n=expected,correct=k,accuracy_percent=100*k/n if n else None,
            wilson_95_percent=[100*x for x in wilson(k,n)] if n else None,
            official_percent=official,difference_official_pp=100*k/n-official if n else None,
            status=('COMPLETE_PILOT_NOT_FULL_BENCHMARK' if a.pilot else 'COMPLETE') if n==expected else 'PARTIAL_NOT_FINAL',
            scope='USER_APPROVED_PILOT' if a.pilot else 'ORIGINAL_FULL_BENCHMARK',
            mean_nfe=statistics.mean(r['nfe'] for r in data),p95_nfe=percentile([r['nfe'] for r in data],95),
            mean_output_tokens=statistics.mean(r['generation_tokens'] for r in data),
            eos=sum(r['termination_reason'] in ('EOS','EOS_FILTERED_SOURCE_DERIVED') for r in data),
            capped=sum(r['termination_reason']=='GEN_LENGTH_CAP' for r in data),
            remaining_mask=sum(r['remaining_mask_tokens']>0 for r in data),
            mean_quality_observer_wall_ms=statistics.mean(r['wall_ms'] for r in data),
            p50_quality_observer_wall_ms=percentile([r['wall_ms'] for r in data],50),
            primary_isolated_latency='NOT_MEASURED')
        summaries.append(summary)
    comparisons=[]
    for task in ('gsm8k','humaneval'):
        hf={r['sample_id']:r for r in samples if r['scoring_method']=='REFERENCE_HF' and r['task']==task}
        ep={r['sample_id']:r for r in samples if r['scoring_method']=='P0_FULL_EP4' and r['task']==task}
        common=sorted(hf.keys() & ep.keys())
        scored=[i for i in common if hf[i]['correct'] is not None and ep[i]['correct'] is not None]
        comparisons.append(dict(task=task,matched_generations=len(common),matched_scored_n=len(scored),
            exact_output_agreement=sum(hf[i]['output_ids']==ep[i]['output_ids'] for i in common),
            accuracy_ep_minus_hf_pp=100*statistics.mean(int(ep[i]['correct'])-int(hf[i]['correct']) for i in scored) if scored else None,
            paired_bootstrap_95_pp=paired_bootstrap([hf[i]['correct'] for i in scored],[ep[i]['correct'] for i in scored]),
            caveat='Different HF multinomial and native greedy decoder protocols. Not a sharding-only attribution.'))
    write(destination/'baseline_quality_summary.json',dict(cohorts=summaries, comparisons=comparisons))
    if summaries:
        with (destination/'baseline_accuracy.csv').open('w') as stream:
            out=csv.DictWriter(stream,fieldnames=list(summaries[0]))
            out.writeheader()
            out.writerows(summaries)
    with (destination/'quality_samples.jsonl').open('w') as stream:
        for r in samples:
            stream.write(json.dumps(r,ensure_ascii=False)+'\n')
    print(json.dumps(dict(cohorts=summaries,comparisons=comparisons),indent=2))

if __name__=='__main__':
    main()
