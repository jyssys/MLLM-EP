"""Fresh TRAIN fixed-policy tuning. The best tested point is not a global oracle."""
import gzip
import json
from .common import ROOT,GPUS,write,digest

def main():
    assert not (ROOT/'p2_fixed_tuning_plan.json').exists()
    data=json.loads(gzip.open(ROOT/'workload_payloads.json.gz','rt').read())['cases']
    groups={d:[r for r in data if r['split']=='TRAIN' and r['domain']==d] for d in ('chat','long','stress8k')}
    scenarios=[]
    for width,domain in [(4,'long'),(8,'stress8k')]:
        for trial in (0,1):
            cases=groups['chat'][trial*(width-1):trial*(width-1)+width-1]+[groups[domain][trial]]
            scenarios.append(dict(id=f'TRAIN_FIXED_N{width}_{domain}_T{trial}',split='TRAIN',workload=domain.upper()+'_INTERFERENCE',
                case_ids=[r['id'] for r in cases],arrivals_s=[0.]*(width-1)+[.25],duration_s=.251,
                design='Finite natural input cohort. Input-only fixed-policy tuning.'))
    manifest=ROOT/'fixed_tuning_scenarios.json';write(manifest,dict(scenarios=scenarios))
    variants=[]
    for C in (128,512,1024,2048):
        for G,runtime in [(1,'split'),(2,'grouped'),(3,'grouped3'),(4,'staged4')]:
            variants.append(dict(label=f'FIX_C{C}_G{G}',runtime=runtime,C=C,fairness='completion'))
        variants.append(dict(label=f'FIX_ATOMIC_C{C}',runtime='atomic',C=C,fairness='completion'))
        variants.append(dict(label=f'FIX_MIXED_C{C}',runtime='mixed',C=C,fairness='completion'))
    jobs=[]
    for index,gpu in enumerate(GPUS):
        width=4 if index%2==0 else 8
        ordered=variants[index:]+variants[:index]
        if index//2:ordered=list(reversed(ordered))
        jobs.append(dict(id=f'dual_p2_FIXED_N{width}_S{index//2}_gpu{gpu}_v1',kind='dual_serving',gpu=gpu,
            width=width,cells=ordered,scenarios=[s['id'] for s in scenarios if s['id'].startswith(f'TRAIN_FIXED_N{width}_')],
            scenario_manifest=manifest.name,scenario_sha256=digest(manifest),cap=1024,budget=2048,blocks=18432,
            graphs=True,streams=True,drain_s=300,timeout_s=10800,lane='LIVE_MEASURED_TRAIN_FIXED_TUNING_SHARED_HOST',
            normalization_sha256=digest(ROOT/'objective_normalization_v1.json'),workload_lock_sha256=digest(ROOT/'workload_lock.json')))
    write(ROOT/'p2_fixed_tuning_plan.json',dict(gpus=GPUS,jobs=jobs,
        scope='24 fixed native points, including Atomic and Mixed. Two starts per width. Strong-control tuning, not FINAL evidence.'))
    print('Registered',len(jobs),'fixed-tuning jobs with',len(variants),'native policy cells')

if __name__=='__main__':main()
