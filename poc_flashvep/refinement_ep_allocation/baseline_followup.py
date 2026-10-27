"""Wait for the quality queue. Then validate isolated A/B baselines, not sparse policies."""
import argparse
import fcntl
import json
import os
import subprocess
import sys
import time
from .common import ROOT, REPO, EP_PYTHON, setup, write, append
from .campaign import environment
from . import resources

def alive(pid):
    path=__import__('pathlib').Path(f'/proc/{pid}/stat')
    try:
        return path.read_text().split(') ',1)[1][0]!='Z'
    except FileNotFoundError:
        return False

def execute(name,command,gpus):
    with (ROOT/'logs'/f'{name}.log').open('a') as log:
        child=subprocess.Popen(command,cwd=REPO,env=environment(gpus),stdout=log,
            stderr=subprocess.STDOUT,start_new_session=True)
    append(ROOT/'commands.jsonl',dict(name=name,command=command,gpus=gpus,pid=child.pid,time=time.time()))
    write(ROOT/'state/followup.json',dict(status='RUNNING_BASELINE_ONLY',pid=os.getpid(),
        child=child.pid,name=name,time=time.time()))
    code=child.wait()
    append(ROOT/'job_events.jsonl',dict(name=name,pid=child.pid,exit_code=code,time=time.time()))
    if code:
        raise RuntimeError(f'{name} exited with code {code}. See its preserved log.')

def run():
    setup()
    lock=(ROOT/'state/followup.lock').open('a')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    write(ROOT/'state/followup.json',dict(status='WAITING_FOR_FULL_BASELINE_QUEUE',pid=os.getpid(),time=time.time()))
    while True:
        state=json.loads((ROOT/'state/campaign.json').read_text())
        if state['status']=='BASELINE_QUEUE_ENDED_REVIEW_REQUIRED' and not alive(state['pid']):
            break
        time.sleep(15)
    result=subprocess.run([sys.executable,'-m','poc_flashvep.refinement_ep_allocation.postprocess','--score-humaneval'],
        cwd=REPO,capture_output=True,text=True)
    (ROOT/'logs/baseline_scoring.log').write_text(result.stdout+result.stderr)
    assert result.returncode==0,'Baseline scoring failed. Preserve its log.'
    summary=json.loads((ROOT/'baseline_quality_summary.json').read_text())
    required={('REFERENCE_HF','gsm8k'),('REFERENCE_HF','humaneval'),
              ('P0_FULL_EP4','gsm8k'),('P0_FULL_EP4','humaneval')}
    complete={(r['method'],r['task']) for r in summary['cohorts'] if r['status']=='COMPLETE'}
    if not required<=complete:
        write(ROOT/'state/followup.json',dict(status='BLOCKED_INCOMPLETE_BASELINE',
            missing=sorted(required-complete),time=time.time()))
        return
    # These runs retain all edges. They do not approve algorithmic optimization.
    resources.base.stop(list(range(8)))
    try:
        for group,start in [('A',0),('B',0),('B',1),('A',1)]:
            gpus=list(range(4)) if group=='A' else list(range(4,8))
            name=f'{group}_isolated_start{start}'
            status=ROOT/'ep4'/name/'status.json'
            if status.exists() and json.loads(status.read_text())['status']=='COMPLETE':
                continue
            execute(name,[EP_PYTHON,'-m','poc_flashvep.refinement_ep_allocation.ep4_worker',
                '--manifest',str(ROOT/'manifests/validation.jsonl'),'--gpus',','.join(map(str,gpus)),
                '--run-id',name,'--limit','10','--parity','--repeats','3','--timing-kind','ISOLATED_EP4'],gpus)
        for group in ('A','B'):
            gpus=list(range(4)) if group=='A' else list(range(4,8))
            name=f'{group}_validation_observer'
            status=ROOT/'ep4'/name/'status.json'
            if status.exists() and json.loads(status.read_text())['status']=='COMPLETE':
                continue
            execute(name,[EP_PYTHON,'-m','poc_flashvep.refinement_ep_allocation.ep4_worker',
                '--manifest',str(ROOT/'manifests/validation.jsonl'),'--gpus',','.join(map(str,gpus)),
                '--run-id',name,'--limit','2','--parity','--trace'],gpus)
        write(ROOT/'state/followup.json',dict(status='BASELINE_COMPLETE_AWAIT_PHASE2_REVIEW',time=time.time(),
            next='Review full quality and observer overhead. Apply sensitivity and EP headroom gates before sparse engineering.'))
    except BaseException as error:
        write(ROOT/'state/followup.json',dict(status='FAILED_BASELINE_FOLLOWUP',error=repr(error),time=time.time()))
        raise
    finally:
        resources.base.restore(list(range(8)))

def main():
    setup()
    p=argparse.ArgumentParser()
    p.add_argument('action',choices=('start','run'))
    a=p.parse_args()
    if a.action=='run':
        run()
    else:
        with (ROOT/'logs/followup_supervisor.log').open('a') as log:
            child=subprocess.Popen([sys.executable,'-m','poc_flashvep.refinement_ep_allocation.baseline_followup','run'],
                cwd=REPO,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        print('FOLLOWUP_STARTED',child.pid)

if __name__=='__main__':
    main()
