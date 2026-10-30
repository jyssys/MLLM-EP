"""Detached, bounded continuation. Never turn an unfinished gate into a pass."""
import argparse
import json
import os
import subprocess
import time
from pathlib import Path
from .common import *

MODULE='poc_flashvep.adaptive_loop_token_pd_2gpu.'

def command(module,*args,check=True):
    cmd=[str(PYTHON),'-m',MODULE+module,*args]
    append(ROOT/'controller_commands.jsonl',dict(time=time.time(),command=cmd))
    result=subprocess.run(cmd,cwd=REPO)
    if check and result.returncode:raise RuntimeError(f'Controller substep failed: {module} {args}')
    return result.returncode

def other_dispatchers():
    found=[]
    for path in Path('/proc').iterdir():
        if not path.name.isdecimal():continue
        try:
            words=(path/'cmdline').read_bytes().split(b'\0')
            if path.stat().st_uid==os.getuid() and (MODULE+'dispatch').encode() in words:found.append(int(path.name))
        except OSError:pass
    return found

def qualify():
    from .gate import compare
    eligibility={}
    for model in ('ouro','huginn'):
        reference=f'qualification32_{model}_pd_g1_v1'
        for arm in ARMS:
            name=f'qualification32_{model}_{arm.lower()}_v1'
            status=ROOT/'raw'/name/'status.json'
            try:
                if not status.exists() or json.loads(status.read_text())['status']!='COMPLETE':raise RuntimeError('Qualification job is not complete')
                result=compare(reference,name)
            except Exception as error:result=dict(passed=False,error=repr(error),candidate=name)
            write(ROOT/'state'/f'gate_{name}.json',result);eligibility[f'{model}_{arm}']=result['passed']
    write(ROOT/'state/qualification_eligibility.json',eligibility)
    return eligibility

def preflight():
    cal=json.loads((ROOT/'slo_calibration.json').read_text());full=0.;small=0.;tune=0.
    for key,c in cal.items():
        model,w=key.split('_');duration=8/(c['rate']/{'W1':.8,'W2':1.,'W3':.8,'W4':1.1}[w])
        for counts,which in ((dict(W1=32,W2=24,W3=32,W4=24),'full'),(dict(W1=24,W2=16,W3=24,W4=16),'small')):
            estimate=7*(50+(counts[w]-1)/c['rate']+.6*duration)
            if which=='full':full+=estimate
            else:small+=estimate
        tune+=2*3*(50/4+duration*.5)
    reserve=4500.;available=remaining()-reserve
    limited=full+tune>available*.8
    reduced=full+tune>available
    plan=dict(time=time.time(),main_full_estimate_s=full,main_reduced_estimate_s=small,
        limited_tuning_estimate_s=tune,limited_static_grid=limited,reduced_matrix=reduced,
        remaining_s=remaining(),reserve_s=reserve,
        estimation='Planning estimate from disjoint PD_G1 calibration. Not measured candidate speedup.',
        tuning_server_estimate_s=max(300,tune/4))
    write(ROOT/'state/preflight_plan.json',plan);budget(selected_matrix='BUDGET_REDUCED_MATRIX' if reduced else 'FULL_ONE_START_MATRIX',projection=plan)
    return plan

def run():
    setup();write(ROOT/'state/controller.json',dict(pid=os.getpid(),status='WAITING_FOR_QUALIFICATION',time=time.time()))
    while other_dispatchers():
        if remaining()<4000:return
        time.sleep(5)
    if (ROOT/'configs/primary_sequence.json').exists():
        command('primary','run',check=False);command('analyze')
        write(ROOT/'state/controller.json',dict(pid=os.getpid(),status='PRIMARY_FINISHED_NEEDS_REVIEW',time=time.time()))
        return
    eligible=qualify()
    if not any(eligible.get(f'{m}_PD_G1') for m in ('ouro','huginn')):
        write(ROOT/'state/controller.json',dict(status='BLOCKED_CORRECTNESS',time=time.time()));return
    jobs=[f'calibration_{m}_{w.lower()}_pd_g1.json' for m in ('ouro','huginn')
          if eligible.get(f'{m}_PD_G1') for w in ('W1','W2','W3','W4')]
    write(ROOT/'state/controller.json',dict(pid=os.getpid(),status='SLO_CALIBRATION',time=time.time()))
    command('dispatch',*jobs,'--quiet',check=False)
    if not (ROOT/'slo_calibration.json').exists():command('calibration','freeze')
    plan_path=ROOT/'state/preflight_plan.json'
    plan=json.loads(plan_path.read_text()) if plan_path.exists() else preflight()
    if not list((ROOT/'configs').glob('tuning_*_v1.json')):command('static_tuning','make')
    jobs=[p.name for p in sorted((ROOT/'configs').glob('tuning_*_v1.json'))
          if eligible.get(f'{json.loads(p.read_text())["model"]}_{json.loads(p.read_text())["arm"]}')]
    write(ROOT/'state/controller.json',dict(pid=os.getpid(),status='STATIC_CALIBRATION',time=time.time()))
    command('dispatch',*jobs,'--quiet',check=False)
    if not (ROOT/'static_selection.json').exists():command('static_tuning','select')
    command('primary','make',*(['--reduced'] if plan['reduced_matrix'] else []))
    write(ROOT/'state/controller.json',dict(pid=os.getpid(),status='PRIMARY_MATRIX',time=time.time()))
    command('primary','run',check=False)
    command('analyze')
    write(ROOT/'state/controller.json',dict(pid=os.getpid(),status='PRIMARY_FINISHED_NEEDS_REVIEW',time=time.time()))
    from .resources import base as resources
    resources.restore(list(range(8)))

def main():
    p=argparse.ArgumentParser();p.add_argument('action',choices=['start','run']);a=p.parse_args();setup()
    if a.action=='run':return run()
    previous=ROOT/'state/controller_supervisor.json'
    if previous.exists():
        pid=json.loads(previous.read_text())['pid']
        process=Path('/proc')/str(pid)
        try:
            if process.stat().st_uid==os.getuid() and (MODULE+'campaign').encode() in (process/'cmdline').read_bytes().split(b'\0'):
                raise RuntimeError(f'A campaign controller is already active: {pid}')
        except FileNotFoundError:pass
    with (ROOT/'logs/controller.log').open('a') as handle:
        child=subprocess.Popen([str(PYTHON),'-m',MODULE+'campaign','run'],cwd=REPO,
            stdout=handle,stderr=subprocess.STDOUT,start_new_session=True)
    write(ROOT/'state/controller_supervisor.json',dict(pid=child.pid,start_time=time.time()))
    print('DETACHED_BOUNDED_CONTROLLER',child.pid)

if __name__=='__main__':main()
