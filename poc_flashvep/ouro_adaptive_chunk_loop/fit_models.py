"""Freeze validation selection before any blind-test metric is computed."""
import csv
import gzip
import json
import time
from collections import defaultdict
import numpy as np
from .common import ROOT, write, digest
from . import predictor as q
from . import decode_prediction as d

SPLITS=('TRAIN','VALIDATION','CALIBRATION','HELDOUT')

def dataset():
    quantum=[]
    decode=[]
    families={s:set() for s in SPLITS}
    plan=json.loads((ROOT/'p2_surface_plan.json').read_text())
    assert all((ROOT/'raw'/job['id']/'completed.json').exists() for job in plan['jobs']), 'MAIN_GRID_INCOMPLETE'
    sources=[]
    for job in plan['jobs']:
        folder=ROOT/'raw'/job['id']
        for path in sorted(folder.glob('*_T*.json')):
            data=json.loads(path.read_text())
            sources.append(dict(path=str(path.relative_to(ROOT)),sha256=digest(path)))
            for row in data['quantum_observations']:
                split=row['split']
                families[split].update(row['source_families'])
                # Retain only causal features and separately marked labels.
                f=q.feature_dict(row)
                compact=dict(f,spans=[dict(phase=s['phase'],tokens=s['tokens']) for s in row['spans']],
                    policy=row['policy'],split=split,run_id=row['run_id'],trial=row['trial'],
                    source_families=row['source_families'],physical_gpu=row['physical_gpu'],P=row['P'],
                    configured_C=row['configured_C'],schema=row.get('observer_schema_version',1),
                    recurrent_depth_start=row['recurrent_depth_start'],target_ms=row['scheduling_impact_ms'],
                    device_interval_ms=row['device_stream_interval_ms'],enqueue_ms=row['enqueue_to_ready_ms'])
                # feature_dict recomputes actual_C from native spans.
                quantum.append(compact)
                for r in d.expand([row]):
                    r.update(P=row['P'],physical_gpu=row['physical_gpu'],schema=row.get('observer_schema_version',1),trial=row['trial'])
                    decode.append(r)
    for a in SPLITS:
        for b in SPLITS:
            assert a==b or not families[a]&families[b], (a,b,'SOURCE_FAMILY_LEAKAGE')
    write(ROOT/'predictor_dataset_sources.json',dict(files=sources,counts=dict(quantum=len(quantum),decode=len(decode)),
        families={s:sorted(v) for s,v in families.items()},scope='Diagnostic stock-policy event lane. Not clean serving.'))
    return quantum,decode

def validation_score(rows,expected):
    by_run=defaultdict(list)
    for row in rows:
        by_run[row['run_id']].append(abs(row['target_ms']-expected(row)))
    return float(np.mean([np.mean(v) for v in by_run.values()]))

def metrics(rows,expected,margin):
    truth=np.array([r['target_ms'] for r in rows]);p=np.array([expected(r) for r in rows]);error=truth-p
    runs=defaultdict(list)
    for i,row in enumerate(rows):
        runs[row['run_id']].append(i)
    totals=np.array([[len(ix),np.abs(error[ix]).sum(),(error[ix]<=margin).sum()] for ix in runs.values()])
    rng=np.random.default_rng(20261023)
    draws=totals[rng.integers(0,len(totals),size=(2000,len(totals)))].sum(axis=1)
    ci_mae=np.quantile(draws[:,1]/draws[:,0],[.025,.975]).tolist()
    ci_cover=np.quantile(draws[:,2]/draws[:,0],[.025,.975]).tolist()
    started=time.perf_counter_ns()
    for _ in range(3):
        for row in rows:
            expected(row)
    overhead=(time.perf_counter_ns()-started)/(3*len(rows))
    return dict(observations=len(rows),process_groups=len(runs),MAE_ms=float(np.mean(abs(error))),MAE_CI95_ms=ci_mae,
        RMSE_ms=float(np.sqrt(np.mean(error**2))),p90_positive_error_ms=float(np.quantile(np.maximum(error,0),.90)),
        p95_positive_error_ms=float(np.quantile(np.maximum(error,0),.95)),p99_positive_error_ms=float(np.quantile(np.maximum(error,0),.99)),
        underprediction_fraction=float(np.mean(error>0)),safe_margin_ms=margin,
        upper_bound_coverage=float(np.mean(error<=margin)),coverage_CI95=ci_cover,query_overhead_us=overhead/1000)

def export(rows,name):
    # Nested causal spans/source IDs use JSON within quoted CSV cells.
    fields=list(dict.fromkeys(k for row in rows for k in row))
    with gzip.open(ROOT/name,'wt') as stream:
        writer=csv.DictWriter(stream,fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({k:json.dumps(v) if isinstance(v,(dict,list,tuple)) else v for k,v in row.items()})

def decode_fit(kind,train):
    groups=defaultdict(list)
    for r in train:
        groups[d.key(r)].append(r['target_ms'])
    table={k:float(np.median(v)) for k,v in groups.items()}
    mean=float(np.mean([r['target_ms'] for r in train]))
    coefficient=None
    if kind=='D2':
        x=np.array([d.vector(r) for r in train]);y=np.array([r['target_ms'] for r in train])
        coefficient=np.linalg.solve(x.T@x+.01*np.eye(x.shape[1]),x.T@y)
    artifact=dict(kind=kind,mean=mean,coefficient=None if coefficient is None else coefficient.tolist(),
        table=[dict(key=k,value=v) for k,v in table.items()])
    if kind=='D0':
        expected=lambda row:mean
    elif kind=='D1':
        expected=lambda row:table.get(d.key(row),mean)
    else:
        expected=lambda row:max(.001,float(np.dot(d.vector(row),coefficient)))
    return expected,artifact

def main():
    protocol=ROOT/'predictor_protocol.json'
    registration=json.loads(protocol.read_text())
    output=ROOT/'predictor_blind_results.json'
    assert not output.exists(),'BLIND_RESULTS_EXIST. No automatic re-fit or overwrite.'
    quantum,decode=dataset()
    models={}
    report={}
    for target,rows,prefix in [('quantum',quantum,'E'),('next_completed_token',decode,'D')]:
        groups={s:[r for r in rows if r['split']==s] for s in SPLITS}
        assert all(groups.values())
        fitted={}
        validation=[]
        for index in range(3):
            name=f'{prefix}{index}'
            if prefix=='E':
                model=q.fit(name,groups['TRAIN'])
                expected=model.expected
                artifact=dict(kind=name,mean=model.global_mean,coefficient=None if model.coefficient is None else model.coefficient.tolist(),
                    table=[dict(key=k,value=v) for k,v in model.table.items()])
            else:
                expected,artifact=decode_fit(name,groups['TRAIN'])
            score=validation_score(groups['VALIDATION'],expected)
            fitted[name]=(expected,artifact)
            validation.append(dict(model=name,process_group_mean_MAE_ms=score))
        best=min(r['process_group_mean_MAE_ms'] for r in validation)
        selected=next(r['model'] for r in validation if r['process_group_mean_MAE_ms']<=1.05*best)
        for name,(expected,artifact) in fitted.items():
            margin=max(0.,float(np.quantile([r['target_ms']-expected(r) for r in groups['CALIBRATION']],.99)))
            artifact['upper_margin_ms']=margin
            models[f'{target}_{name}']=artifact
        selection=dict(target=target,selected=selected,validation=validation,protocol_sha256=digest(protocol),
            model=models[f'{target}_{selected}'],feature_source_sha256=digest(q.__file__ if prefix=='E' else d.__file__),
            frozen_wall_s=time.time(),scope='Selected on VALIDATION. CALIBRATION supplies only residual margin. No HELDOUT metric evaluated yet.')
        # Freeze selected model and margin before calculating test metrics.
        write(ROOT/'predictor_artifact'/f'{target}_selection.json',selection)
        values=[]
        strata=[]
        for name,(expected,artifact) in fitted.items():
            margin=models[f'{target}_{name}']['upper_margin_ms']
            for split in ('VALIDATION','HELDOUT'):
                values.append(dict(model=name,split=split,selected=name==selected,**metrics(groups[split],expected,margin)))
            if name==selected:
                for label,key_fn in [('shape',lambda r:(r['P'],r['N'])),('schema',lambda r:r['schema']),
                                      ('policy',lambda r:r['policy']),('phase',lambda r:q.phase(r) if prefix=='E' else r['phase'])]:
                    bins=defaultdict(list)
                    for row in groups['HELDOUT']:
                        bins[str(key_fn(row))].append(row)
                    for key,subset in bins.items():
                        strata.append(dict(stratum=label,key=key,**metrics(subset,expected,margin)))
        chosen=next(v for v in values if v['selected'] and v['split']=='HELDOUT')
        gate=chosen['upper_bound_coverage']>=registration['initial_coverage_target'] and chosen['query_overhead_us']<=registration['query_budget_us']
        report[target]=dict(selected=selected,validation_selection=validation,results=values,strata=strata,
            provisional_gate='PASS_WITH_LIMITS' if gate else 'BLOCKED_PREDICTOR',
            limitation='Stock-policy retrospective prediction. Candidate-specific rollout and legal queue simulation remain untested.')
        for split in SPLITS:
            export(groups[split],f'{target}_{split.lower()}_dataset.csv.gz')
    write(ROOT/'predictor_artifact'/'all_models.json',models)
    write(output,report)
    print(json.dumps({k:dict(selected=v['selected'],gate=v['provisional_gate'],heldout=next(x for x in v['results'] if x['selected'] and x['split']=='HELDOUT')) for k,v in report.items()},indent=2))

if __name__=='__main__':
    main()
