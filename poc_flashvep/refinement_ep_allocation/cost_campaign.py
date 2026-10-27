"""Run isolated operator cost checks after the semantic screen releases both groups."""
import argparse
import fcntl
import json
import os
import subprocess
import sys
import time
from .common import ROOT,REPO,EP_PYTHON,write,append,setup
from .campaign import launch
from . import resources

def run(clean=False):
    setup()
    statepath=ROOT/('state/cost_clean.json' if clean else 'state/cost.json')
    lock=(ROOT/('state/cost_clean.lock' if clean else 'state/cost.lock')).open('a')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    if clean:
        while json.loads((ROOT/'state/cost.json').read_text())['status']=='ISOLATED_OPERATOR_MEASUREMENT':
            write(statepath,dict(status='WAITING_FOR_BASELINE_QUEUE',pid=os.getpid(),time=time.time()))
            time.sleep(3)
    while True:
        state=json.loads((ROOT/'state/sensitivity.json').read_text())
        if state['status']!='RUNNING':
            break
        write(statepath,dict(status='WAITING_FOR_SEMANTIC_SCREEN',pid=os.getpid(),time=time.time()))
        time.sleep(3)
    if state['status']=='FAILED':
        write(statepath,dict(status='BLOCKED_CONTROL_OR_SCREEN_FAILURE',time=time.time()))
        return
    resources.base.stop(list(range(8)))
    failures=[]
    try:
        for group,gpus in [('A',[0,1,2,3]),('B',[4,5,6,7])]:
            name=f'{group}_clean_operator_cost' if clean else f'{group}_isolated_operator_cost'
            status=ROOT/'ep4'/name/'status.json'
            if status.exists() and json.loads(status.read_text())['status']=='COMPLETE':
                continue
            cmd=[EP_PYTHON,'-m','poc_flashvep.refinement_ep_allocation.ep4_worker','--manifest',
                str(ROOT/'manifests/validation.jsonl'),'--limit','1','--gpus',','.join(map(str,gpus)),
                '--run-id',name,'--operator-probe','--timing-kind','ISOLATED_EP4']
            cmd+=['--repeats','1' if clean else '33']
            child=launch(name,cmd,gpus)
            while child.poll() is None:
                write(statepath,dict(status='ISOLATED_OPERATOR_MEASUREMENT',
                    name=name,pid=os.getpid(),worker=child.pid,time=time.time(),other_group='QUIET'))
                time.sleep(3)
            append(ROOT/'job_events.jsonl',dict(name=name,pid=child.pid,exit_code=child.returncode,time=time.time()))
            if child.returncode:
                failures.append(name)
                break
    finally:
        resources.base.restore(list(range(8)))
    write(statepath,dict(status='FAILED' if failures else 'COMPLETE_AWAIT_COST_ANALYSIS',
        failures=failures,time=time.time()))

def main():
    setup()
    p=argparse.ArgumentParser()
    p.add_argument('action',choices=('run','start','run-clean','start-clean'))
    a=p.parse_args()
    if a.action.startswith('run'):
        run(clean=a.action=='run-clean')
    else:
        with (ROOT/'logs/cost_supervisor.log').open('a') as log:
            child=subprocess.Popen([sys.executable,'-m','poc_flashvep.refinement_ep_allocation.cost_campaign',
                'run-clean' if a.action=='start-clean' else 'run'],
                cwd=REPO,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        print('COST_CONTROLLER',child.pid)

if __name__=='__main__':
    main()
