"""Register balanced validation before reading new live policy outcomes."""
import csv
import gzip
import json
import random
from .common import ROOT,GPUS,write,digest

FAMILIES=[('O2',8.),('O4',1.),('O6',2.),('O10',8.)]

def cells():
    tuning=list(csv.DictReader((ROOT/'fixed_C_objective_tuning.csv').open()))
    result=[]
    for family,weight in FAMILIES:
        options=[r for r in tuning if r['objective']==family and float(r['weight'])==weight]
        fixed=int(min(options,key=lambda r:(float(r['mean_ordinal_regret']),-int(r['fixed_C'])))['fixed_C'])
        for surface,suffix in [('JOINT','J'),('G_ONLY','G'),('C_ONLY','C')]:
            result.append(dict(label=f'{family}_{suffix}',runtime='split',C=2048,surface=surface,
                fixed_C=fixed,objective=family,weight=weight,budget_ms=120.,fairness='completion'))
        # A second large fixed C tests whether the cost-tuned G-only control is weak.
        result.append(dict(label=f'{family}_G1024',runtime='split',C=2048,surface='G_ONLY',
            fixed_C=1024,objective=family,weight=weight,budget_ms=120.,fairness='completion'))
    result.extend([
        dict(label='E_ONLY',runtime='split',C=2048,surface='JOINT',objective='O2',weight=0.,endpoint='E_ONLY',budget_ms=120.),
        dict(label='R_ONLY',runtime='split',C=2048,surface='JOINT',objective='O2',weight=1.,endpoint='R_ONLY',budget_ms=120.),
        dict(label='ATOMIC',runtime='atomic',C=1024,fairness='stock'),
        dict(label='STAGED4',runtime='staged4',C=1024,fairness='completion'),
        dict(label='MIXED_STOCK',runtime='mixed',C=1024,fairness='stock'),
        dict(label='MIXED_FAIR',runtime='mixed',C=1024,fairness='completion')])
    return result

def main():
    assert not (ROOT/'p2_objective_validation_plan.json').exists(),'Preserve frozen validation'
    assert (ROOT/'workload_lock.json').exists()
    data=json.loads(gzip.open(ROOT/'workload_payloads.json.gz','rt').read())['cases']
    val={domain:[r for r in data if r['split']=='VALIDATION' and r['domain']==domain]
        for domain in ('chat','long','stress8k')}
    scenarios=[]
    for width,domain in [(4,'long'),(8,'stress8k')]:
        assert len(val[domain])>=4,(domain,len(val[domain]))
        for trial in range(4):
            cases=val['chat'][trial*(width-1):trial*(width-1)+width-1]+[val[domain][trial]]
            assert len(cases)==width
            scenarios.append(dict(id=f'VAL_N{width}_{domain}_T{trial}',split='VALIDATION',workload=domain.upper()+'_INTERFERENCE',
                case_ids=[r['id'] for r in cases],arrivals_s=[0.]*(width-1)+[.250],duration_s=.251,
                design='Finite burst plus delayed natural long prefill. Not capacity or Poisson serving.'))
    write(ROOT/'serving_scenarios.json',dict(scenarios=scenarios))
    variants=cells();jobs=[]
    for width in (4,8):
        for start,gpu in enumerate(GPUS):
            selected=[dict(c) for c in variants]
            selected=selected[start%len(selected):]+selected[:start%len(selected)]
            if start%2:selected=list(reversed(selected))
            # Every cell crosses every GPU. Context and objective never select a device.
            jobs.append(dict(id=f'dual_p2_VAL_N{width}_S{start}_gpu{gpu}_v1',kind='dual_serving',gpu=gpu,
                width=width,cells=selected,scenarios=[s['id'] for s in scenarios if s['id'].startswith(f'VAL_N{width}_')][start%2::2],
                cap=1024,budget=2048,blocks=18432,graphs=True,streams=True,drain_s=300,timeout_s=10800,
                lane='LIVE_MEASURED_VALIDATION_SHARED_HOST',
                normalization_sha256=digest(ROOT/'objective_normalization_v1.json'),
                workload_lock_sha256=digest(ROOT/'workload_lock.json')))
    write(ROOT/'objective_shortlist_development.json',dict(families=FAMILIES,arms=variants,
        rationale='Historical TRAIN geometry plus distinct mechanisms: coupled sum, risk-class lexicographic, observed age weights, yield costs.',
        primary_control='Same objective, features, fairness, cost universe, and B. Only the permitted action surface changes.',
        secondary_G_control='C1024 supplements the historical TRAIN cost-tuned fixed C; do not hide a stronger G-only result.',
        family_selection_is_not_serving_evidence=True))
    write(ROOT/'p2_objective_validation_plan.json',dict(gpus=GPUS,jobs=jobs,
        scope='Four same-GPU/model-process blocks per width. Two VALIDATION source cohorts per block. Not final evidence.'))
    print('Registered objective validation',len(jobs),'jobs;',len(variants),'arms per same-GPU block')

if __name__=='__main__':main()
