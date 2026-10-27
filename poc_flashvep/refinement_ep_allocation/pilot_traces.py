"""Two independent EP4 replicas collect unmodified validation trajectories."""
import argparse
import fcntl
import os
import subprocess
import sys
import time
from .common import ROOT,REPO,EP_PYTHON,write,append,setup
from .campaign import launch
from . import resources

def run():
    setup()
    lock=(ROOT/'state/pilot_traces.lock').open('a')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    resources.base.stop(list(range(8)))
    jobs={}
    failed=[]
    try:
        for shard,group,gpus in [(0,'A',[0,1,2,3]),(1,'B',[4,5,6,7])]:
            name=f'{group}_pilot_validation_trace'
            jobs[group]=launch(name,[EP_PYTHON,'-m','poc_flashvep.refinement_ep_allocation.ep4_worker',
                '--manifest',str(ROOT/'manifests/validation.jsonl'),'--limit','16','--shards','2',
                '--shard',str(shard),'--gpus',','.join(map(str,gpus)),'--run-id',name,
                '--parity','--trace','--refinement-telemetry'],gpus)
        while jobs:
            for group,child in list(jobs.items()):
                code=child.poll()
                if code is None:
                    continue
                append(ROOT/'job_events.jsonl',dict(name=f'{group}_pilot_validation_trace',pid=child.pid,
                    exit_code=code,time=time.time()))
                del jobs[group]
                if code:
                    failed.append(group)
                resources.base.restore([0,1,2,3] if group=='A' else [4,5,6,7])
            write(ROOT/'state/pilot_traces.json',dict(status='RUNNING' if jobs else 'COMPLETE' if not failed else 'FAILED',
                active={k:v.pid for k,v in jobs.items()},failed=failed,time=time.time(),
                evidence_scope='16 non-test validation requests. Unmodified inference plus diagnostic confidence history.',
                timing_scope='PARALLEL_OBSERVER_NOT_PRIMARY_LATENCY'))
            if jobs:
                time.sleep(5)
    finally:
        if not jobs:
            resources.base.restore(list(range(8)))

def main():
    setup()
    p=argparse.ArgumentParser()
    p.add_argument('action',choices=('start','run'))
    a=p.parse_args()
    if a.action=='run':
        run()
    else:
        with (ROOT/'logs/pilot_traces_supervisor.log').open('a') as log:
            child=subprocess.Popen([sys.executable,'-m','poc_flashvep.refinement_ep_allocation.pilot_traces','run'],
                cwd=REPO,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        print('PILOT_TRACE_CONTROLLER',child.pid)

if __name__=='__main__':
    main()
