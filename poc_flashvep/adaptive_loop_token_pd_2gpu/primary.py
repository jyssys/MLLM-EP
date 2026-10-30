"""Predeclare the 56-cell matrix and execute only independently qualified arms."""
import argparse
import json
import random
import subprocess
import time
from .common import *
from .prepare import base
from .static_tuning import offsets
from .gate import lines

def make(reduced=False):
    calibration=json.loads((ROOT/'slo_calibration.json').read_text())
    choices=json.loads((ROOT/'static_selection.json').read_text())['selected']
    counts=dict(W1=24,W2=16,W3=24,W4=16) if reduced else dict(W1=32,W2=24,W3=32,W4=24)
    sequence=[]
    for wi,w in enumerate(('W1','W2','W3','W4')):
        for mi,model in enumerate(('ouro','huginn')):
            if f'{model}_{w}' not in calibration:continue
            n=counts[w];rows=json.loads((ROOT/'data'/f'{w}_test.json').read_text())
            if reduced and w=='W1':
                # Recheck actual new bin support rather than trusting prior labels.
                rows=sum(([r for r in rows if all(a<=v<b for v in r['input_lengths'].values())][:k]
                    for a,b,k in ((800,1500,8),(1500,2800,10),(2800,3585,6))),[])
                random.Random(2026103144).shuffle(rows)
                caps=[384]*6+[448]*12+[512]*6;random.Random(2026103145).shuffle(caps)
                rows=[dict(r,cap=c) for r,c in zip(rows,caps)]
            elif reduced and w=='W3':
                rows=rows[:9]+rows[12:18]+rows[20:29]
                caps=[256]*6+[384]*12+[512]*6;random.Random(2026103146).shuffle(caps)
                rows=[dict(r,cap=c) for r,c in zip(rows,caps)]
            else:rows=rows[:n]
            if reduced and w in ('W2','W4'):
                caps=([128]*5+[192]*6+[256]*5) if w=='W2' else ([400]*5+[448]*6+[512]*5)
                random.Random(2026103147+wi).shuffle(caps)
                rows=[dict(r,cap=c) for r,c in zip(rows,caps)]
            if len(rows)!=n:raise RuntimeError('Reduced manifest count mismatch')
            dataset=f'primary_{model}_{w}.json';write(ROOT/'data'/dataset,rows)
            cal=calibration[f'{model}_{w}'];arrivals=offsets(n,cal['rate'],2026103140+wi)
            if w=='W3':
                arrivals=[0.];rng=random.Random(2026103140+wi)
                for i in range(1,n):
                    burst=all(v>=2800 for v in rows[i]['input_lengths'].values())
                    arrivals.append(arrivals[-1]+rng.expovariate(cal['rate']*(5 if burst else 1)))
            order=[x for x in ARMS if x!='PD_G1'];random.Random(2026103150+wi+mi).shuffle(order)
            for arm in ['PD_G1']+order:
                # Hardware-match every causal comparison in one workload/model.
                # Cross the pair across workloads and later independent starts.
                pair=(wi+2*mi)%4;gpus=[2*pair,2*pair+1]
                name=f'primary_{model}_{w.lower()}_{arm.lower()}_start0'
                cfg=base(model,arm,name,dataset,n,gpus)
                cfg.update(cal,phase='primary',workload=w,start=0,seed=2026103160+wi,
                    arrival_offsets=arrivals,request_caps=[r['cap'] for r in rows],
                    prompt_ids=[r['prompt_id'] for r in rows],timeout=3600,estimated_seconds=1500,
                    matrix_scope='BUDGET_REDUCED_MATRIX' if reduced else 'FULL_ONE_START_MATRIX')
                axis='token' if arm.startswith('Token') else 'loop'
                selected=choices.get(f'{model}_{w}_{axis}',dict(cut=cfg['depth'],prefix_fraction=1.))
                cfg['static_tuning_status']='CALIBRATED' if f'{model}_{w}_{axis}' in choices else 'FULL_PD_FALLBACK_TUNING_INCOMPLETE'
                if arm.endswith('Static'):cfg.update(cut=selected['cut'],prefix_fraction=selected['prefix_fraction'])
                if arm=='DP':cfg.update(chunk=4096,token_budget=4096,wavefront=False)
                path=ROOT/'configs'/f'{name}.json'
                if path.exists():raise FileExistsError(path)
                write(path,cfg);sequence.append(path.name)
    write(ROOT/'configs/primary_sequence.json',sequence)
    # Transport-only controls follow the mandatory matrix. They do not replace cells.
    for model in ('ouro','huginn'):
        ref=ROOT/'configs'/f'primary_{model}_w1_pd_g1_start0.json'
        if not ref.exists():continue
        cfg=json.loads(ref.read_text());name=f'protocol_{model}_w1_full_snapshot_start0'
        cfg.update(name=name,arm='Loop_Static',axis='loop',dynamic=False,cut=cfg['depth'],
                   phase='protocol_control',handoff_mode='snapshot')
        write(ROOT/'configs'/f'{name}.json',cfg);sequence.append(f'{name}.json')
    write(ROOT/'configs/primary_sequence.json',sequence)
    budget(selected_matrix='BUDGET_REDUCED_MATRIX' if reduced else 'FULL_ONE_START_MATRIX',primary_configurations=56)

def run():
    blocked_path=ROOT/'state/primary_blocked.json'
    blocked=json.loads(blocked_path.read_text()) if blocked_path.exists() else {}
    sequence=json.loads((ROOT/'configs/primary_sequence.json').read_text())
    for name in sequence:
        cfg=json.loads((ROOT/'configs'/name).read_text());key=f'{cfg["model"]}_{cfg["arm"]}'
        qualification=ROOT/'state'/f'gate_qualification32_{cfg["model"]}_{cfg["arm"].lower()}_v1.json'
        if key in blocked or not qualification.exists() or not json.loads(qualification.read_text())['passed']:
            append(ROOT/'state/primary_skipped.jsonl',dict(name=cfg['name'],reason='BLOCKED_CORRECTNESS',time=time.time()))
            continue
        if cfg['arm']!='PD_G1':
            ref=ROOT/'raw'/cfg.get('reference_run',f'primary_{cfg["model"]}_{cfg["workload"].lower()}_pd_g1_start0')/'status.json'
            if not ref.exists() or json.loads(ref.read_text())['status']!='COMPLETE':
                append(ROOT/'state/primary_skipped.jsonl',dict(name=cfg['name'],reason='BLOCKED_CORRECTNESS_REFERENCE',time=time.time()))
                continue
        status=ROOT/'raw'/cfg['name']/'status.json'
        if status.exists() and json.loads(status.read_text())['status']=='COMPLETE':continue
        result=subprocess.run([str(PYTHON),'-m','poc_flashvep.adaptive_loop_token_pd_2gpu.dispatch',name,'--quiet','--keep-quiet'],cwd=REPO)
        subprocess.run([str(PYTHON),'-m','poc_flashvep.adaptive_loop_token_pd_2gpu.analyze'],cwd=REPO,check=True)
        if not status.exists():continue  # The dispatcher recorded the budget skip.
        if result.returncode:
            blocked[key]='FAILED_JOB';write(blocked_path,blocked);continue
        # Controlled same-output comparison is a hard expansion gate.
        if cfg['arm']!='PD_G1':
            reference=ROOT/'raw'/f'primary_{cfg["model"]}_{cfg["workload"].lower()}_pd_g1_start0'
            a={r['sample_id']:r for r in lines(reference/'requests.jsonl') if not r['warmup']}
            b={r['sample_id']:r for r in lines(ROOT/'raw'/cfg['name']/'requests.jsonl') if not r['warmup']}
            checks={i:bool(i in a and all(a[i][k]==r[k] for k in ('input_ids','output_ids','exit_depths','finish_reason'))) for i,r in b.items()}
            value=dict(name=cfg['name'],reference=reference.name,checks=checks,passed=len(a)==len(b)==cfg['count'] and all(checks.values()))
            write(ROOT/'state'/f'primary_parity_{cfg["name"]}.json',value)
            if not value['passed']:blocked[key]='OUTPUT_MISMATCH'
        write(ROOT/'state/primary_blocked.json',blocked)
    from .resources import base as resources
    resources.restore(list(range(8)))
    from .secondary import run as secondary
    secondary()
    from .profiles import capture
    capture()

def main():
    p=argparse.ArgumentParser();p.add_argument('action',choices=['make','run']);p.add_argument('--reduced',action='store_true');a=p.parse_args()
    if a.action=='make':make(a.reduced)
    else:run()

if __name__=='__main__':main()
