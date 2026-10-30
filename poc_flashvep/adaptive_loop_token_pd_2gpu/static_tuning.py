"""Equal calibration cohorts for all static cuts. No held-out selection."""
import argparse
import json
import random
from .common import *
from .prepare import base

def offsets(count,rate,seed):
    rng=random.Random(seed);values=[0.]
    for i in range(1,count):values.append(values[-1]+rng.expovariate(rate))
    return values

def make():
    from .fairness_gate import qualify_final_token_grid
    final_huginn_token_static_ready=qualify_final_token_grid()
    calibrated=json.loads((ROOT/'slo_calibration.json').read_text())
    plan=json.loads((ROOT/'state/preflight_plan.json').read_text()) if (ROOT/'state/preflight_plan.json').exists() else {}
    for model in ('ouro','huginn'):
        for axis in ('loop','token'):
            if model=='huginn' and axis=='token' and not final_huginn_token_static_ready:continue
            arm='Loop_Static' if axis=='loop' else 'Token_Static'
            name=f'tuning_{model}_{axis}_v1'
            cfg=base(model,arm,name,'W1_calibration.json',4,[0,1])
            cfg.update(tune_jobs=[],timeout=3600,estimated_seconds=plan.get('tuning_server_estimate_s',1800))
            for wi,w in enumerate(('W1','W2','W3','W4')):
                if f'{model}_{w}' not in calibrated:continue
                cuts=[0,1,2,3,4] if model=='ouro' else [0,8,16,24,28,32]
                choices=cuts if axis=='loop' else [c/MODEL_PINS[model][2] for c in cuts]
                if plan.get('limited_static_grid'):
                    choices=[0,MODEL_PINS[model][2]//2,MODEL_PINS[model][2]] if axis=='loop' else [0.,.5,1.]
                random.Random(2026103190+wi).shuffle(choices)
                rows=json.loads((ROOT/'data'/f'{w}_calibration.json').read_text())[:4]
                for choice in choices:
                    cohort=len(cfg['tune_jobs'])
                    cut=int(choice) if axis=='loop' else cfg['depth']
                    job=dict(cohort=cohort,workload=w,choice=choice,cut=cut,
                        prefix_fraction=float(choice) if axis=='token' else 1.,count=4,
                        dataset=f'{w}_calibration.json',arrival_offsets=offsets(4,calibrated[f'{model}_{w}']['rate'],2026103194+wi),
                        request_caps=[r['cap'] for r in rows],prompt_ids=[r['prompt_id'] for r in rows],
                        seed=2026103198+wi,**calibrated[f'{model}_{w}'])
                    cfg['tune_jobs'].append(job)
            if cfg['tune_jobs']:write(ROOT/'configs'/f'{name}.json',cfg)

def select():
    selected={}
    details=[]
    for model in ('ouro','huginn'):
        for axis in ('loop','token'):
            name=f'tuning_{model}_{axis}_v1';folder=ROOT/'raw'/name
            status=folder/'status.json'
            if not status.exists() or json.loads(status.read_text())['status']!='COMPLETE':continue
            rows=[json.loads(s) for s in (folder/'requests.jsonl').read_text().splitlines() if s]
            cfg=json.loads((folder/'arc_contract.json').read_text())
            for w in ('W1','W2','W3','W4'):
                choices=[]
                for job in cfg['tune_jobs']:
                    if job['workload']!=w:continue
                    x=[r for r in rows if not r['warmup'] and r.get('cohort')==job['cohort']]
                    if len(x)!=job['count']:raise RuntimeError('Incomplete static tuning cohort')
                    passed=sum(r['ttft_ms']<=job['ttft_slo_ms'] and r['tpot_ms']<=job['tpot_slo_ms'] for r in x)
                    window=json.loads((folder/f'window_{job["cohort"]}.json').read_text())
                    goodput=passed/(window['end']-window['start'])
                    mean=sum(r['e2e_ms'] for r in x)/len(x)
                    choice=dict(choice=job['choice'],cut=job['cut'],prefix_fraction=job['prefix_fraction'],
                        passed=passed,N=len(x),goodput=goodput,mean_e2e_ms=mean,cohort=job['cohort'])
                    choices.append(choice);details.append(dict(model=model,axis=axis,workload=w,**choice))
                if not choices:continue
                best=min(choices,key=lambda r:(-r['passed']/r['N'],-r['goodput'],r['mean_e2e_ms']))
                selected[f'{model}_{w}_{axis}']=best
    path=ROOT/'static_selection.json'
    if path.exists():raise FileExistsError(path)
    write(path,dict(selected=selected,all_calibration_choices=details,
                   selection_rule='Highest joint SLO pass fraction, then cohort goodput, then mean E2E.',
                   scope='Limited disjoint four-request tuning. Not an optimality proof.'))

def main():
    p=argparse.ArgumentParser();p.add_argument('action',choices=['make','select']);a=p.parse_args();globals()[a.action]()

if __name__=='__main__':main()
