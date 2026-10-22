"""Small offline estimators. Future labels never enter feature extraction."""
from dataclasses import dataclass
import json
import math
import time
import numpy as np
from .common import ROOT, write

FEATURE_NAMES=('actual_C','actual_G','context_max','actual_total_rows','actual_decode_rows','N','graph_entries','staged_bytes')
PATHS=('atomic','split','grouped','grouped3','staged4','mixed')

def phase(row):
    values={s['phase'] for s in row['spans']}
    if 'prefill' in values:
        return 'atomic_prefill'
    if 'prefilling_recurrent' in values:
        return 'mixed' if 'recurrent' in values else 'staged_prefill'
    if 'prefilling_prelude' in values:
        return 'prefill_prelude'
    return next(iter(values)) if len(values)==1 else 'other'

def feature_dict(row):
    actual_C=sum(s['tokens'] for s in row['spans'] if s['phase'].startswith('prefill'))
    return dict(actual_C=actual_C,actual_G=row['actual_G'],context_max=row['context_max'],
        actual_total_rows=row['actual_total_rows'],actual_decode_rows=row['actual_decode_rows'],
        N=row['N'],graph_entries=row['graph_entries'],staged_bytes=row['staged_bytes'])

def vector(row):
    f=feature_dict(row)
    c,g,l,b,d,n,graph,stage=(float(f[k]) for k in FEATURE_NAMES)
    return [1.,c/1024,g,c*g/1024,math.log2(max(1,l)),b/1024,d,n,graph/32,stage/2**30]+[
        float(row['policy']==p) for p in PATHS]+[float(phase(row)==p) for p in ['atomic_prefill','staged_prefill','prefill_prelude','recurrent','coda','prelude','mixed']]

def key(row):
    f=feature_dict(row)
    def bucket(x):
        return int(math.ceil(math.log2(max(1,x))))
    return (row['policy'],phase(row),bucket(f['actual_C']),f['actual_G'],bucket(f['context_max']),bucket(f['actual_total_rows']),bucket(f['N']))

@dataclass
class Predictor:
    kind:str
    global_mean:float
    table:dict
    coefficient:object=None
    upper_margin:float=0.

    def expected(self,row):
        if self.kind=='E0':
            return self.global_mean
        if self.kind=='E1':
            return self.table.get(key(row),self.global_mean)
        return max(.001,float(np.dot(vector(row),self.coefficient)))

    def safe(self,row):
        return self.expected(row)+max(0.,self.upper_margin)

    def OOD(self,row):
        return key(row) not in self.table

def fit(kind,train):
    groups={}
    for row in train:
        groups.setdefault(key(row),[]).append(row['target_ms'])
    table={k:float(np.median(v)) for k,v in groups.items()}
    model=Predictor(kind,float(np.mean([r['target_ms'] for r in train])),table)
    if kind=='E2':
        x=np.asarray([vector(r) for r in train]);y=np.asarray([r['target_ms'] for r in train])
        model.coefficient=np.linalg.solve(x.T@x+.01*np.eye(x.shape[1]),x.T@y)
    return model

def calibrate(model,rows):
    residual=[r['target_ms']-model.expected(r) for r in rows]
    model.upper_margin=max(0.,float(np.quantile(residual,.99)))

def assess(model,rows):
    y=np.array([r['target_ms'] for r in rows]);p=np.array([model.expected(r) for r in rows])
    e=y-p
    t=time.perf_counter_ns()
    for _ in range(10):
        for row in rows:
            model.expected(row);model.safe(row)
    overhead=(time.perf_counter_ns()-t)/(10*len(rows))
    return dict(model=model.kind,N=len(rows),MAE_ms=float(np.mean(abs(e))),RMSE_ms=float(np.sqrt(np.mean(e**2))),
        p95_positive_error_ms=float(np.quantile(np.maximum(0,e),.95)),p99_positive_error_ms=float(np.quantile(np.maximum(0,e),.99)),
        underprediction_fraction=float(np.mean(e>0)),upper_bound_coverage=float(np.mean(y<=p+model.upper_margin)),
        safe_margin_ms=model.upper_margin,OOD_fraction=float(np.mean([model.OOD(r) for r in rows])),
        per_expected_plus_safe_query_ns=overhead,
        note='Quantum target diagnostic interval. Does not establish request-level deadline attainment.')

def extract():
    rows=[]
    for folder in (ROOT/'raw').glob('p2_*'):
        if not (folder/'completed.json').exists():
            continue
        for file in folder.glob('*_T*.json'):
            d=json.loads(file.read_text())
            for row in d.get('quantum_observations',[]):
                rows.append(dict(row,target_ms=row['scheduling_impact_ms']))
    return rows

def main():
    rows=extract()
    partitions={s:[r for r in rows if r['split']==s] for s in ['TRAIN','VALIDATION','CALIBRATION','HELDOUT']}
    assert all(partitions.values()),'All disjoint source/run partitions are required'
    # A quantum can contain several families. Every family remains in one split.
    family_sets={s:{f for r in v for f in r['source_families']} for s,v in partitions.items()}
    for a in family_sets:
        for b in family_sets:
            if a!=b:
                assert not family_sets[a]&family_sets[b],(a,b,'SOURCE_FAMILY_LEAKAGE')
    summaries=[]
    for kind in ('E0','E1','E2'):
        model=fit(kind,partitions['TRAIN'])
        calibrate(model,partitions['CALIBRATION'])
        for split in ('VALIDATION','HELDOUT'):
            summaries.append(dict(split=split,**assess(model,partitions[split])))
        write(ROOT/'predictor_artifact'/f'{kind}.json',dict(kind=kind,features=FEATURE_NAMES,
            coefficient=None if model.coefficient is None else model.coefficient.tolist(),
            global_mean=model.global_mean,table=[dict(key=k,value=v) for k,v in model.table.items()],margin=model.upper_margin,
            allowed_features='Only data known before the measured quantum. No completion label, future duration, or future output.'))
    write(ROOT/'predictor_results.json',summaries)
    print(json.dumps(summaries,indent=2))

if __name__=='__main__':
    main()
