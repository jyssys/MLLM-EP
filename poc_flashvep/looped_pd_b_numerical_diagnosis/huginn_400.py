"""One finite H400 gate. Two baseline starts use independent physical pairs."""
from concurrent.futures import ThreadPoolExecutor
import json
import os
import time
from .common import ROOT,setup,write
from .run import execute
from .qualification import compare
from .resources import base


def main():
    setup();state=ROOT/'state/huginn_400_qualification.json'
    while True:
        path=ROOT/'state/ouro_down_norm_qualification.json'
        if path.exists() and json.loads(path.read_text()).get('status') in ('FINITE_QUALIFICATION_PASS','NO_GO_CORRECTNESS','BLOCKED_RUNTIME'):break
        write(state,dict(status='WAITING_FOR_OURO_PAIR',pid=os.getpid(),time=time.time()))
        time.sleep(10)
    time.sleep(3)
    devices=[0,1,6,7]
    try:
        base.stop(devices)
        jobs=[dict(name=f'huginn_down_norm_qual_N400_closed_loop_{arm}_{index}',model='huginn',variant='down_norm',
              arm=arm,workload='diagnostic',dataset='W1_repair_cycles',count=400,cap=128,rate=0.,seed=2026109001,
              gpus=[0,1] if index!=1 else [6,7],capacity=8,timeout=3600,diagnostic=True)
              for index,arm in enumerate(('PD_G1','PD_G1','B'))]
        write(state,dict(status='RUNNING_BASELINE_REPEATS',jobs=jobs,pid=os.getpid(),time=time.time()))
        with ThreadPoolExecutor(max_workers=2) as pool:
            outcomes=list(pool.map(execute,jobs[:2]))
        if any(r.get('status')!='COMPLETE' for r in outcomes):
            write(state,dict(status='BLOCKED_RUNTIME',results=outcomes,time=time.time()));return
        repeated=compare([r['name'] for r in jobs[:2]])
        if not all(r['N']==400 and r['exact']==400 and r['complete'] for r in repeated):
            write(state,dict(status='NO_GO_CORRECTNESS',baseline=repeated,time=time.time()));return
        write(state,dict(status='RUNNING_B',baseline=repeated,time=time.time()))
        value=execute(jobs[2])
        if value.get('status')!='COMPLETE':
            write(state,dict(status='BLOCKED_RUNTIME',result=value,time=time.time()));return
        paired=compare([r['name'] for r in jobs])
        passed=all(r['N']==400 and r['exact']==400 and r['complete'] for r in paired)
        write(state,dict(status='FINITE_QUALIFICATION_PASS' if passed else 'NO_GO_CORRECTNESS',
                         runs=[r['name'] for r in jobs],comparisons=paired,time=time.time(),
                         scope='400-request closed-loop correctness, not open-loop capacity. Baseline starts cross physical pairs.'))
    finally:
        base.restore(devices)


if __name__=='__main__':main()
