"""Register missing live budget bins as CALIBRATION diagnostics only.

This does not retune or replace the completed FINAL comparison.
"""
import copy
import gzip
import json
import sys
from .common import ROOT,write,digest

def main():
    assert not (ROOT/'p2_budget_supplement_plan.json').exists()
    final=json.loads((ROOT/'p3_characterization_plan.json').read_text())['jobs'][0]
    controls=[c for c in final['cells'] if c['label'] in ('O6_J','O6_G','O6_G1024','O6_C','MIXED_FAIR')]
    payload=json.loads(gzip.open(ROOT/'workload_payloads.json.gz','rt').read())['cases']
    pool={d:[c for c in payload if c['split']=='CALIBRATION' and c['domain']==d] for d in ('chat','long','stress8k')}
    scenarios=[];jobs=[]
    for i,B in enumerate((20.,80.,240.)):
        for j,(width,domain) in enumerate(((4,'long'),(8,'stress8k'))):
            gpu=(4,5,6,7)[(2*i+j)%4];name=f'CAL_BUDGET_{int(B)}_N{width}'
            cases=pool['chat'][:width-1]+[pool[domain][i%len(pool[domain])]]
            scenarios.append(dict(id=name,split='CALIBRATION',workload=f'{domain.upper()}_BUDGET_{int(B)}MS',
                case_ids=[c['id'] for c in cases],arrivals_s=[0.]*(width-1)+[.25],duration_s=.251,
                scope='Secondary bounded natural calibration cohort. Not a new FINAL selection dataset or qualified capacity.'))
            cells=copy.deepcopy(controls);cells=cells[i+j:]+cells[:i+j]
            for c in cells:
                if c.get('surface'):c['budget_ms']=B
            jobs.append(dict(id=f'dual_p2_BUDGET_B{int(B)}_N{width}_gpu{gpu}_v1',kind='dual_serving',gpu=gpu,
                width=width,cells=cells,scenarios=[name],scenario_manifest='budget_supplement_scenarios.json',
                cap=1024,budget=2048,blocks=18432,graphs=True,streams=True,drain_s=600,timeout_s=7200,
                lane='LIVE_MEASURED_SECONDARY_BUDGET_CALIBRATION_NOT_FINAL',
                normalization_sha256=digest(ROOT/'objective_normalization_v1.json'),workload_lock_sha256=digest(ROOT/'workload_lock.json')))
    manifest=ROOT/'budget_supplement_scenarios.json';write(manifest,dict(scenarios=scenarios))
    for job in jobs:job['scenario_sha256']=digest(manifest)
    write(ROOT/'p2_budget_supplement_plan.json',dict(gpus=[4,5,6,7],jobs=jobs,
        scope='Six exploratory budget/context starts. One start per cell, no confirmatory CI. Fill initially registered20/80/240ms live coverage, without retuning the fixed120ms primary objective.'))
    print('Registered six secondary budget jobs. Execute only after the quality dispatcher releases its lock.')

def observed_amendment():
    original=ROOT/'p2_budget_supplement_plan.json';plan=json.loads(original.read_text())
    preserved=ROOT/'p2_budget_supplement_unlaunched_v1_plan.json'
    assert not preserved.exists()
    assert all(not (ROOT/'raw'/j['id']).exists() and not (ROOT/'jobs'/f'{j["id"]}.json').exists() for j in plan['jobs'])
    old_hash=digest(original);write(preserved,plan);assert digest(preserved)==old_hash
    for job in plan['jobs']:
        job['id']=job['id'].removesuffix('_v1')+'_v2'
        job['observe']=True
        job['lane']='LIVE_MEASURED_SECONDARY_BUDGET_EVENT_DIAGNOSTIC_NOT_CLEAN_FINAL'
    plan['scope']='Six one-start-per-cell CUDA-event CALIBRATION diagnostics. Includes observer overhead. No clean serving speedup or capacity claim.'
    write(original,plan)
    write(ROOT/'budget_supplement_amendment.json',dict(original_unlaunched_plan=preserved.name,
        original_sha256=old_hash,active_plan_sha256=digest(original),
        changes='New v2 IDs and CUDA-event observations before any supplementary job launches. No primary FINAL change.',
        reason='Measure actual selected-quantum duration and budget overruns in the missing20/80/240ms diagnostic bins. Observer timing is not headline clean serving.'))
    print('Preserved the unlaunched v1 plan. Registered observed v2 diagnostics before GPU execution.')

if __name__=='__main__':
    observed_amendment() if '--observe-amendment' in sys.argv else main()
