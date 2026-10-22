"""Full-next-output-token targets, separate from one recurrent quantum."""
import json
import math
import time
import numpy as np
from .common import ROOT,write
from .predictor import extract,PATHS

def expand(rows):
    results=[]
    for row in rows:
        for residual in row['completed_token_residuals']:
            if residual['last_completed_token_s'] is None:
                continue # First-token TTFT is a separate endpoint.
            results.append(dict(run_id=row['run_id'],source_families=row['source_families'],split=row['split'],policy=row['policy'],
                request_id=residual['id'],phase=residual['phase'],remaining_R=residual['remaining_R'],
                context=row['context_max'],N=row['N'],active_requests=len(row['state']),
                pending_prefill_tokens=sum(max(0,s['prompt_tokens']-s['num_prefilled']) for s in row['state']),
                staged_prompt_tokens=sum(max(0,s.get('num_preluded',s['num_prefilled'])-s['num_prefilled']) for s in row['state']),
                configured_C=row['configured_C'],actual_G=row['actual_G'],graph_entries=row['graph_entries'],
                pending_submission_events=row.get('pending_submission_events',0),
                host_load1=row.get('host_load1',0.),load_missing=int('host_load1' not in row),
                elapsed_since_last_token_ms=residual['time_since_last_token_ms'],
                target_ms=residual['time_to_completed_next_token_ms']))
    return results

def key(row):
    return (row['policy'],row['phase'],row['remaining_R'],int(math.ceil(math.log2(max(1,row['context'])))),row['N'])

def vector(row):
    return [1.,row['remaining_R'],math.log2(max(1,row['context'])),row['N'],row['active_requests'],row['elapsed_since_last_token_ms']/1000,
        row['pending_prefill_tokens']/8192,row['staged_prompt_tokens']/2048,row['configured_C']/1024,row['actual_G'],
        row['graph_entries']/32,row['host_load1']/8,row['load_missing'],row.get('pending_submission_events',0)]+[
        float(row['phase']==p) for p in ('prelude','recurrent','coda')]+[float(row['policy']==p) for p in PATHS]

def main():
    rows=expand(extract())
    groups={s:[r for r in rows if r['split']==s] for s in ('TRAIN','VALIDATION','CALIBRATION','HELDOUT')}
    assert all(groups.values()),'All disjoint data partitions are required'
    train=groups['TRAIN']
    table={}
    for row in train:
        table.setdefault(key(row),[]).append(row['target_ms'])
    lookup={k:float(np.median(v)) for k,v in table.items()}
    mean=float(np.mean([r['target_ms'] for r in train]))
    x=np.array([vector(r) for r in train]);y=np.array([r['target_ms'] for r in train])
    coefficient=np.linalg.solve(x.T@x+.01*np.eye(x.shape[1]),x.T@y)
    predictions=[('D0',lambda r:mean),('D1',lambda r:lookup.get(key(r),mean)),
        ('D2',lambda r:max(.001,float(np.dot(vector(r),coefficient))))]
    reports=[]
    for name,predict in predictions:
        margin=max(0.,float(np.quantile([r['target_ms']-predict(r) for r in groups['CALIBRATION']],.99)))
        for split in ('VALIDATION','HELDOUT'):
            selected=groups[split]
            truth=np.array([r['target_ms'] for r in selected]);p=np.array([predict(r) for r in selected]);e=truth-p
            started=time.perf_counter_ns()
            for row in selected:
                predict(row)
            reports.append(dict(model=name,split=split,observations=len(selected),
                process_groups=len({r['run_id'] for r in selected}),source_families=len({f for r in selected for f in r['source_families']}),
                MAE_ms=float(np.mean(abs(e))),RMSE_ms=float(np.sqrt(np.mean(e**2))),
                underprediction_fraction=float(np.mean(e>0)),p95_positive_error_ms=float(np.quantile(np.maximum(0,e),.95)),
                p99_positive_error_ms=float(np.quantile(np.maximum(0,e),.99)),safe_margin_ms=margin,
                upper_bound_coverage=float(np.mean(e<=margin)),query_overhead_ns=(time.perf_counter_ns()-started)/len(selected),
                scope='Retrospective stock-policy full-token prediction. New-policy rollout generalization is untested.'))
    write(ROOT/'decode_completion_predictor_results.json',reports)
    write(ROOT/'predictor_artifact'/'decode_models.json',dict(mean=mean,linear_coefficient=coefficient.tolist(),
        lookup=[dict(key=k,median=v) for k,v in lookup.items()],
        forbidden_inputs=['future_token_ids','future_output_length','future_arrivals','future_prefill_interruptions','actual_next_token_time']))
    print(json.dumps(reports,indent=2))

if __name__=='__main__':
    main()
