"""Run phase-2 free-running interventions. Never report these timings as acceleration."""
import argparse
import fcntl
import json
import os
import subprocess
import sys
import time
from .common import ROOT,REPO,EP_PYTHON,setup,write,append
from .campaign import launch
from . import resources

def command(group,kind,fraction,control=False):
    gpus=[0,1,2,3] if group=='A' else [4,5,6,7]
    name=f'{group}_semantic_{kind}_{int(fraction*100):02d}'
    if kind=='P1':
        name+='_uniform_v2'
    if control:
        name='A_semantic_zero_v2' if group=='A' else 'B_semantic_zero_control'
    cmd=[EP_PYTHON,'-m','poc_flashvep.refinement_ep_allocation.ep4_worker','--manifest',
        str(ROOT/'manifests/validation.jsonl'),'--limit','16','--run-id',name,
        '--gpus',','.join(map(str,gpus)),'--semantic-emulation',kind,'--edge-fraction',str(fraction)]
    if control:
        cmd+=['--shards','2','--shard','0' if group=='A' else '1']
    return name,cmd,gpus

def run_batch(queues,stage):
    running={}
    failures=[]
    while queues or running:
        for group,queue in list(queues.items()):
            if group in running:
                continue
            while queue:
                item=queue.pop(0)
                name,cmd,gpus=item
                status=ROOT/'ep4'/name/'status.json'
                if status.exists() and json.loads(status.read_text())['status']=='COMPLETE':
                    continue
                if status.exists() and json.loads(status.read_text())['status']=='RUNNING':
                    saved=json.loads(status.read_text())
                    pid=saved['pid']
                    command_line=open(f'/proc/{pid}/cmdline','rb').read().replace(b'\0',b' ').decode()
                    assert os.stat(f'/proc/{pid}').st_uid==os.getuid() and name in command_line
                    class Adopted:
                        def __init__(self,pid,path):
                            self.pid=pid
                            self.path=path
                        def poll(self):
                            state=json.loads(self.path.read_text())['status']
                            return None if state=='RUNNING' else 0 if state=='COMPLETE' else 1
                    child=Adopted(pid,status)
                else:
                    resources.base.stop(gpus)
                    child=launch(name,cmd,gpus)
                running[group]=(name,child,gpus)
                break
            if not queue:
                del queues[group]
        for group,(name,child,gpus) in list(running.items()):
            code=child.poll()
            if code is None:
                continue
            append(ROOT/'job_events.jsonl',dict(name=name,pid=child.pid,exit_code=code,time=time.time()))
            del running[group]
            if code:
                failures.append(name)
                queues.pop(group,None)
            if group not in queues:
                resources.base.restore(gpus)
        write(ROOT/'state/sensitivity.json',dict(status='RUNNING',stage=stage,pid=os.getpid(),
            active={g:dict(name=n,pid=p.pid) for g,(n,p,_) in running.items()},failures=failures,time=time.time(),
            evidence='FREE_RUNNING_DENSE_SEMANTIC_EMULATION_NOT_PHYSICAL_SPEEDUP'))
        if queues or running:
            time.sleep(3)
    return failures

def run():
    setup()
    lock=(ROOT/'state/sensitivity.lock').open('a')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    controls={g:[command(g,'P2',0.,True)] for g in ('A','B')}
    failed=run_batch(controls,'ZERO_EDGE_CONTROL')
    if not failed:
        queues={'A':[command('A','P2',.20),command('A','P1',.05),command('A','P3',.05)],
                'B':[command('B','P2',.05),command('B','P3',.20),command('B','P1',.20)]}
        failed=run_batch(queues,'P1_P2_P3_QUALITY_WORK_SCREEN')
    resources.base.restore(list(range(8)))
    write(ROOT/'state/sensitivity.json',dict(status='FAILED' if failed else 'COMPLETE_AWAIT_QUALITY_AND_COST_REVIEW',
        failures=failed,time=time.time(),
        next='Score on separate validation answers. Fit physical EP costs. Do not count emulated omission as actual work reduction.'))

def main():
    setup()
    p=argparse.ArgumentParser()
    p.add_argument('action',choices=('start','run'))
    a=p.parse_args()
    if a.action=='run':
        run()
    else:
        with (ROOT/'logs/sensitivity_supervisor.log').open('a') as log:
            child=subprocess.Popen([sys.executable,'-m','poc_flashvep.refinement_ep_allocation.sensitivity_campaign','run'],
                cwd=REPO,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        print('SENSITIVITY_CONTROLLER',child.pid)

if __name__=='__main__':
    main()
