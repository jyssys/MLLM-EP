"""Fresh per-arm compile caches for the storage counterfactual."""
import argparse
import datetime as dt
import os
import subprocess
import time
from poc_ouro_parallelism_deep_discovery.contract import *


def stamp():return dt.datetime.now(dt.timezone.utc).isoformat()


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--P',type=int,choices=[2,4],default=2)
    args=parser.parse_args()
    target=ROOT/('artifacts/storage_isolated_queue_v2.json' if args.P==2 else 'artifacts/storage_isolated_P4_queue_v2.json')
    assert not target.exists()
    state=dict(pid=os.getpid(),start_utc=stamp(),status='WAITING_FOR_OWN_PREREQUISITE',jobs=[],
               reason='The v1 arms shared a native compilation cache. Keep v1 evidence but do not use it for a residency conclusion.',
               waiting_is_substantive_research=False)
    write(target,state)
    prerequisite=ROOT/('outputs/storage_alias_base14_P4_v1/guard_status.json' if args.P==2 else 'outputs/storage_alias_base26_P2_isolated_v2/guard_status.json')
    deadline=time.monotonic()+3600
    while not prerequisite.exists():
        if time.monotonic()>deadline:raise TimeoutError(str(prerequisite))
        time.sleep(1)
    assert read(prerequisite).get('success')
    P=args.P;gpus='4,6' if P==2 else '4,5,6,7'
    for size,arm in [('1.4B','alias'),('1.4B','copies'),('2.6B','copies'),('2.6B','alias')]:
        short=size.replace('.','').replace('B','')
        name=f'storage_{arm}_base{short}_P{P}_isolated_v2'
        command=[REFERENCE_PYTHON,'-m','poc_ouro_parallelism_deep_discovery.scripts.gpu_session',
                 '--gpus',gpus,'--output','outputs/'+name,'--timeout','3600','--',
                 PYTHON,'-m','poc_ouro_parallelism_deep_discovery.scripts.native_baseline',
                 '--size',size,'--tp',str(P),'--R','4','--async-scheduling',
                 '--weight-storage-control',arm,'--isolate-compile-cache']
        job=dict(name=name,command=command,start_utc=stamp(),status='RUNNING')
        state['jobs'].append(job);state['status']='RUNNING';write(target,state)
        result=subprocess.run(command,cwd=PROJECT)
        job.update(exit_code=result.returncode,end_utc=stamp(),status='PASS' if result.returncode==0 else 'FAIL')
        write(target,state)
        if result.returncode:
            state.update(status='STOPPED_AFTER_FAILURE',end_utc=stamp());write(target,state)
            raise SystemExit(result.returncode)
    state.update(status='COMPLETE',end_utc=stamp());write(target,state)


if __name__=='__main__':main()
