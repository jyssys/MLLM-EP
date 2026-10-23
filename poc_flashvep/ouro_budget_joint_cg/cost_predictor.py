"""Fresh path-conditioned service lookup. No future-token deadline target."""
import argparse
import csv
import gzip
import json
import math
import time
from collections import defaultdict
from functools import lru_cache
import numpy as np
from .common import ROOT,write,digest

CONTEXTS=(512,2048,8192,16384,65536)

def context_bin(value):
    return next((n for n in CONTEXTS if value<=n),65536)

def path_features(row):
    spans=row['spans'];phases={s['phase'] for s in spans}
    if phases=={'prefill'}:
        path='atomic'
    elif phases=={'prefilling_prelude'}:
        path='staged_prelude'
    elif phases=={'prefilling_recurrent'}:
        path='staged_core'
    elif 'prefilling_recurrent' in phases:
        path='mixed_core'
    else:
        path='decode_'+next(iter(phases)) if len(phases)==1 else 'decode_mixed'
    prefills=[s for s in spans if s['phase'].startswith('prefill')]
    return dict(path=path,rows=row['actual_rows'],G=max((s['G'] for s in prefills),default=1),
        depth=min((s['depth'] for s in prefills),default=0),context=row['context_max'],
        context_bin=context_bin(row['context_max']),prompt_rows=len(prefills),
        decode_rows=len(spans)-len(prefills))

def collect():
    records=[]
    for folder in sorted((ROOT/'raw').glob('cb_p2_*')):
        if not (folder/'completed.json').exists():
            continue
        identity=json.loads((folder/'identity.json').read_text())
        for file in folder.glob('*_T*.json'):
            data=json.loads(file.read_text())
            for row in data.get('service_observations',[]):
                f=path_features(row)
                prompt=[s for s in row['spans'] if s['phase'].startswith('prefill')]
                families=data.get('families',[])
                family=families[int(prompt[0]['id'])] if len(prompt)==1 and families else 'MULTIPLE_OR_DECODE'
                records.append(dict(**f,service_ms=row['service_ms'],split=data['split'],
                    source_family=family,
                    run_id=data['run_id'],GPU=identity['physical_GPU'],trial=data['trial'],
                    policy=data['policy'],configured_C=data['C'],P=data['P'],N=data['configured_N'],
                    host_dispatch_ms=row['host_dispatch_ms'],queue_delay_ms=row['enqueue_to_service_start_ms'],
                    pending_events=sum(not value for value in row['pending_event_ready']),
                    all_requests_ready=all(row['ready_before_issue'].values()),
                    graph_captures=row['graph_after'].get('captures',0)-row['graph_before'].get('captures',0),
                    graph_replays=row['graph_after'].get('replays',0)-row['graph_before'].get('replays',0),
                    staged_bytes=row['staged_bytes'],completed_prompt_tokens=row['completed_prompt_tokens'],
                    prompt_token_loop_work=row['prompt_token_loop_work'],
                    evidence='MEASURED_OBSERVER_READY_STREAM_SERVICE_NOT_PURE_KERNEL_BUSY'))
    return records

class ServiceLookup:
    def __init__(self,artifact):
        self.artifact=artifact;self.tables=artifact['tables'];self.slack=artifact.get('slack_ms',{})
        self.by_path=defaultdict(list)
        for item in self.tables:
            self.by_path[(item['path'],item['G'])].append(item)

    def raw(self,path,c,g,context,depth):
        return self.cached(path,c,g,context_bin(context),depth)

    @lru_cache(maxsize=16384)
    def cached(self,path,c,g,target,depth):
        choices=self.by_path.get((path,g),[])
        if not choices:
            # Unsupported shapes are explicit. This deliberately expensive
            # fallback can only support experimental progress, not a bound.
            candidates=[r for r in self.tables if r['path']==path]
            if not candidates:
                return 10000.,False
            return 1.5*max(r['q95_ms'] for r in candidates),False
        exact_depth=[r for r in choices if r['depth']==depth]
        choices=exact_depth or choices
        def distance(r):
            # Prefer empirical enclosing rows/context over extrapolation.
            return (r['rows']<c,r['context_bin']<target,
                abs(math.log2(max(1,r['rows'])/max(1,c))),abs(math.log2(r['context_bin']/target)))
        chosen=min(choices,key=distance)
        supported=chosen['depth']==depth and chosen['rows']==c and chosen['context_bin']==target
        base=chosen['median_ms']
        if chosen['rows']<c or chosen['context_bin']<target:
            base=max(base,chosen['q95_ms'])*1.3
        return float(base),supported

    def estimate(self,path,c,g,context,depth):
        base,known=self.raw(path,c,g,context,depth)
        return base+self.slack.get(path,0.),known

def fit(records):
    grouped=defaultdict(list)
    for row in records:
        if row['split']!='TRAIN' or row['prompt_rows']!=1 or row['decode_rows']:
            continue
        grouped[(row['path'],row['rows'],row['G'],row['depth'],row['context_bin'])].append(row['service_ms'])
    tables=[]
    for key,values in sorted(grouped.items()):
        p,c,g,d,context=key
        tables.append(dict(path=p,rows=c,G=g,depth=d,context_bin=context,n=len(values),
            median_ms=float(np.median(values)),q95_ms=float(np.percentile(values,95)),maximum_ms=max(values)))
    assert tables,'Fresh TRAIN service samples are required'
    artifact=dict(version='PATH_LOOKUP_V1',tables=tables,slack_ms={},
        target='Native ready-stream incremental service. CPU launch gaps may remain. Not kernel-only busy time.',
        registration='Nearest enclosing measured rows/context. Train medians plus calibration positive residual P95. No old E1 or D1.',
        unsupported='Unknown shapes use explicit inflated empirical fallback. No proven upper bound.')
    lookup=ServiceLookup(artifact);errors=defaultdict(list)
    for row in records:
        if row['split']=='CALIBRATION' and row['prompt_rows']==1 and not row['decode_rows']:
            predicted,_=lookup.raw(row['path'],row['rows'],row['G'],row['context'],row['depth'])
            errors[row['path']].append(max(0.,row['service_ms']-predicted))
    artifact['slack_ms']={key:float(np.percentile(values,95)) for key,values in errors.items()}
    artifact['sample_counts']={split:sum(r['split']==split for r in records)
        for split in ('TRAIN','VALIDATION','CALIBRATION','NEW_HELDOUT')}
    return artifact

def evaluate(records,artifact):
    lookup=ServiceLookup(artifact);residuals=[];groups=defaultdict(list)
    for row in records:
        if row['prompt_rows']!=1 or row['decode_rows']:
            continue
        predicted,known=lookup.raw(row['path'],row['rows'],row['G'],row['context'],row['depth'])
        upper,_=lookup.estimate(row['path'],row['rows'],row['G'],row['context'],row['depth'])
        result=dict(row,predicted_ms=predicted,upper_ms=upper,known_shape=known,
            signed_error_ms=row['service_ms']-predicted,over_upper=row['service_ms']>upper)
        residuals.append(result);groups[(row['split'],row['path'])].append(result)
    summary=[]
    for (split,path),values in sorted(groups.items()):
        errors=np.array([r['signed_error_ms'] for r in values])
        blocks=defaultdict(list)
        for row in values:
            blocks[row['run_id']].append(row['over_upper'])
        counts=np.array([len(v) for v in blocks.values()]);misses=np.array([sum(v) for v in blocks.values()])
        rng=np.random.default_rng(20261023)
        draws=rng.integers(0,len(counts),size=(2000,len(counts)))
        over=100*misses[draws].sum(axis=1)/counts[draws].sum(axis=1)
        summary.append(dict(split=split,path=path,n=len(values),starts=len({r['run_id'] for r in values}),
            source_families=len({r['source_family'] for r in values}),
            MAE_ms=float(np.abs(errors).mean()),median_signed_ms=float(np.median(errors)),
            signed_P95_ms=float(np.percentile(errors,95)),signed_P99_ms=float(np.percentile(errors,99)),
            underprediction_pct=100*float((errors>0).mean()),upper_overrun_pct=100*np.mean([r['over_upper'] for r in values]),
            overrun_block_CI95_low=float(np.percentile(over,2.5)),overrun_block_CI95_high=float(np.percentile(over,97.5)),
            unknown_shape_pct=100*np.mean([not r['known_shape'] for r in values])))
    return residuals,summary

def csv_file(path,rows):
    if not rows:
        return
    opener=gzip.open if str(path).endswith('.gz') else open
    with opener(path,'wt',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--fit',action='store_true');args=parser.parse_args()
    rows=collect();csv_file(ROOT/'cost_service_raw.csv.gz',rows)
    if not args.fit:
        print('Collected',len(rows),'service observations');return
    artifact=fit(rows);target=ROOT/'cost_model_artifacts/service_lookup_v1.json'
    assert not target.exists(),'Frozen model cannot be overwritten'
    write(target,artifact)
    residuals,summary=evaluate(rows,artifact)
    csv_file(ROOT/'prediction_residuals.csv.gz',residuals);csv_file(ROOT/'predictor_heldout.csv',summary)
    write(ROOT/'cost_model_artifacts/freeze_v1.json',dict(sha256=digest(target),created_wall_s=time.time(),
        blind_policy='No model revision after this held-out evaluation. FINAL_ONLINE remains unused.'))
    classifications=[]
    for split in ('VALIDATION','NEW_HELDOUT'):
        for path in ('atomic','staged_prelude','staged_core'):
            for budget in (10,20,40,60,80,120,160,240):
                feasible=[r for r in residuals if r['split']==split and r['path']==path and r['upper_ms']<=budget]
                classifications.append(dict(split=split,path=path,budget_ms=budget,predicted_feasible=len(feasible),
                    false_feasible=sum(r['service_ms']>budget for r in feasible),
                    evidence='OFFLINE_MEASURED_SAMPLE_CLASSIFICATION_NOT_LIVE_COUNTERFACTUAL_SPEED'))
    csv_file(ROOT/'predictor_false_feasible.csv',classifications)
    lookup=ServiceLookup(artifact);queries=[]
    states=[(r['path'],r['rows'],r['G'],r['context'],r['depth']) for r in residuals if r['split']=='VALIDATION'][:1000]
    for repeat in range(3):
        for state in states:
            before=time.perf_counter_ns();lookup.estimate(*state);queries.append(time.perf_counter_ns()-before)
    write(ROOT/'cost_model_artifacts/query_overhead.json',dict(queries=len(queries),P50_us=float(np.percentile(queries,50))/1000,
        P99_us=float(np.percentile(queries,99))/1000,scope='CPU model query only, repeated validation states. Not full selector or GPU service.'))
    print(json.dumps(summary,indent=2))

if __name__=='__main__':
    main()
