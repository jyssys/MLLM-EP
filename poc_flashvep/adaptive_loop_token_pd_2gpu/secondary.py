"""Predeclared W1 knee repeats and an optional W3 rate point."""
import json
import random
import subprocess
from .common import *
from .static_tuning import offsets
from .gate import lines
from .analyze import metric

ARMS_SECONDARY=['PD_G1','DP','Loop_Static','Loop_Dynamic','Token_Dynamic']
PLAN=[dict(workload='W1',rate_multiplier=1.5,start=i) for i in (1,2,3)]
PLAN += [dict(workload='W3',rate_multiplier=1.5,start=1)]
SECONDARY_REQUESTS=16

def run():
    blocked_path=ROOT/'state/secondary_blocked.json'
    blocked=json.loads(blocked_path.read_text()) if blocked_path.exists() else {}
    planned=ROOT/'configs/secondary_predeclared_plan.json'
    if not planned.exists():write(planned,dict(groups=PLAN,arms=ARMS_SECONDARY,
        selection='Declared before held-out primary results. Never select a favorable workload after candidate measurement.',
        priority='W1 three independent starts at1.5 times the frozen base rate, then one W3 rate point.',
        warning='Base-rate start0 is not pooled with1.5-rate starts for uncertainty.'))
    contract=ROOT/'configs/secondary_repeat_budget_contract.json'
    if not contract.exists():write(contract,dict(requests_per_cell=SECONDARY_REQUESTS,
        scope='SECONDARY_BUDGET_LIMITED_N16. Mandatory primary matrix counts remain unchanged.',
        declaration_time=time.time(),reason='Preserve three independent starts within the12-hour global budget.',
        restriction='Do not pool these16-request cohorts with the primary32-request or24-request cohorts.'))
    for number,group in enumerate(PLAN):
        jobs=[];expected=0.
        for mi,model in enumerate(('ouro','huginn')):
            w=group['workload'];start=group['start'];factor=group['rate_multiplier']
            ref_name=f'repeat_{model}_{w.lower()}_pd_g1_rate15_start{start}'
            order=ARMS_SECONDARY[1:];random.Random(2026103200+number+mi).shuffle(order)
            for arm in ['PD_G1']+order:
                if f'{model}_{arm}' in blocked or f'{model}_PD_G1' in blocked:continue
                source=ROOT/'configs'/f'primary_{model}_{w.lower()}_{arm.lower()}_start0.json'
                if not source.exists():continue
                cfg=json.loads(source.read_text());directory=ROOT/'raw'/cfg['name']
                if not (directory/'status.json').exists() or json.loads((directory/'status.json').read_text())['status']!='COMPLETE':continue
                result,_=metric(cfg,directory)
                if result['status']!='COMPLETE':continue
                name=f'repeat_{model}_{w.lower()}_{arm.lower()}_rate15_start{start}'
                original_count=cfg['count']
                dataset=ROOT/'data'/cfg['dataset'];all_rows=json.loads(dataset.read_text())
                if w=='W1':
                    selected=sum(([r for r in all_rows if all(a<=v<b for v in r['input_lengths'].values())][:k]
                        for a,b,k in ((800,1500,6),(1500,2800,6),(2800,3585,4))),[])
                    random.Random(2026103600).shuffle(selected);caps=[384]*4+[448]*8+[512]*4
                else:
                    normal=[r for r in all_rows if not all(v>=2800 for v in r['input_lengths'].values())]
                    long=[r for r in all_rows if all(v>=2800 for v in r['input_lengths'].values())]
                    selected=normal[:6]+long[:4]+normal[6:12];caps=[256]*4+[384]*8+[512]*4
                if len(selected)!=SECONDARY_REQUESTS:raise RuntimeError('Secondary full-text subset is incomplete')
                random.Random(2026103601).shuffle(caps)
                selected=[dict(r,cap=c) for r,c in zip(selected,caps)]
                subset=f'secondary16_{model}_{w}.json';subset_path=ROOT/'data'/subset
                if not subset_path.exists():write(subset_path,selected)
                cfg.update(name=name,phase='repeat',start=start,seed=2026103300+10*number,
                    reference_run=ref_name,rate=cfg['rate']*factor,
                    rate_multiplier=factor,base_rate=json.loads(source.read_text())['rate'],
                    count=SECONDARY_REQUESTS,dataset=subset,request_caps=caps,prompt_ids=[r['prompt_id'] for r in selected],
                    matrix_scope='SECONDARY_BUDGET_LIMITED_N16')
                cfg['arrival_offsets']=offsets(cfg['count'],cfg['rate'],2026103400+number)
                if w=='W3':
                    rows=json.loads((ROOT/'data'/cfg['dataset']).read_text());rng=random.Random(2026103400+number)
                    values=[0.]
                    for row in rows[1:]:
                        burst=all(v>=2800 for v in row['input_lengths'].values())
                        values.append(values[-1]+rng.expovariate(cfg['rate']*(5 if burst else 1)))
                    cfg['arrival_offsets']=values
                pair=(start+2*mi+('W3'==w)*2)%4;cfg['gpus']=[2*pair,2*pair+1]
                cfg['estimated_seconds']=1.25*(result.get('cohort_makespan_s',600)*SECONDARY_REQUESTS/original_count/factor+90)
                path=ROOT/'configs'/f'{name}.json'
                if not path.exists():write(path,cfg)
                done=ROOT/'raw'/name/'status.json'
                if not done.exists():expected+=cfg['estimated_seconds']
                jobs.append((path,cfg))
        if not jobs:continue
        if remaining()<expected+3600:
            for _,cfg in jobs:
                if not (ROOT/'raw'/cfg['name']).exists():
                    append(ROOT/'state/secondary_skipped.jsonl',dict(name=cfg['name'],reason='NOT_RUN_TIME_BUDGET',group=group,time=time.time()))
            continue
        write(ROOT/'state/secondary_progress.json',dict(group=group,status='RUNNING',expected_seconds=expected,time=time.time()))
        for path,cfg in jobs:
            status=ROOT/'raw'/cfg['name']/'status.json'
            if status.exists():continue
            if f'{cfg["model"]}_{cfg["arm"]}' in blocked or f'{cfg["model"]}_PD_G1' in blocked:continue
            if cfg['arm']!='PD_G1':
                reference=ROOT/'raw'/cfg['reference_run']/'status.json'
                if not reference.exists() or json.loads(reference.read_text())['status']!='COMPLETE':continue
            result=subprocess.run([str(PYTHON),'-m','poc_flashvep.adaptive_loop_token_pd_2gpu.dispatch',path.name,'--quiet','--keep-quiet'],cwd=REPO)
            if result.returncode:
                blocked[f'{cfg["model"]}_{cfg["arm"]}']='FAILED_SECONDARY_JOB'
                write(blocked_path,blocked);continue
            if not status.exists():continue
            candidate=ROOT/'raw'/cfg['name']
            if cfg['arm']!='PD_G1':
                reference=ROOT/'raw'/cfg['reference_run']
                a={r['sample_id']:r for r in lines(reference/'requests.jsonl') if not r['warmup']}
                b={r['sample_id']:r for r in lines(candidate/'requests.jsonl') if not r['warmup']}
                checks={i:i in a and all(a[i][k]==r[k] for k in ('input_ids','output_ids','exit_depths','finish_reason')) for i,r in b.items()}
                write(ROOT/'state'/f'primary_parity_{cfg["name"]}.json',dict(passed=len(a)==len(b)==cfg['count'] and all(checks.values()),checks=checks))
            subprocess.run([str(PYTHON),'-m','poc_flashvep.adaptive_loop_token_pd_2gpu.analyze'],cwd=REPO,check=True)
        write(ROOT/'state/secondary_progress.json',dict(group=group,status='GROUP_FINISHED',time=time.time()))
    from .resources import base as resources
    resources.restore(list(range(8)))

if __name__=='__main__':run()
