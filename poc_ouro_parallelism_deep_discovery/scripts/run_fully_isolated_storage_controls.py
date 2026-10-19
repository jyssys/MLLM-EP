"""Complete cache isolation and compiled-operand checks for storage controls."""
import argparse
import datetime as dt
import os
import subprocess
from poc_ouro_parallelism_deep_discovery.contract import *


def stamp():return dt.datetime.now(dt.timezone.utc).isoformat()


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--attempt',default='v3')
    args=parser.parse_args();assert args.attempt.replace('_','').isalnum()
    target=ROOT/f'artifacts/storage_fully_isolated_queue_{args.attempt}.json';assert not target.exists()
    state=dict(pid=os.getpid(),start_utc=stamp(),status='RUNNING',jobs=[],
               reason='v2 left the global AOT model cache shared. No v1/v2 storage-effect conclusion is valid.',
               scope='Counterfactual only, not a candidate method or natural quality result.')
    write(target,state)
    for P in [2,4]:
        gpus='4,6' if P==2 else '4,5,6,7'
        for size,arm in [('1.4B','alias'),('1.4B','copies'),('2.6B','copies'),('2.6B','alias')]:
            short=size.replace('.','').replace('B','')
            name=f'storage_{arm}_base{short}_P{P}_fully_isolated_{args.attempt}'
            command=[REFERENCE_PYTHON,'-m','poc_ouro_parallelism_deep_discovery.scripts.gpu_session',
                     '--gpus',gpus,'--output','outputs/'+name,'--timeout','3600','--',PYTHON,
                     '-m','poc_ouro_parallelism_deep_discovery.scripts.native_baseline',
                     '--size',size,'--tp',str(P),'--R','4','--async-scheduling',
                     '--weight-storage-control',arm,'--isolate-all-compile-caches']
            job=dict(name=name,command=command,start_utc=stamp(),status='RUNNING')
            state['jobs'].append(job);write(target,state)
            result=subprocess.run(command,cwd=PROJECT)
            job.update(exit_code=result.returncode,end_utc=stamp(),status='PASS' if result.returncode==0 else 'FAIL')
            write(target,state)
            if result.returncode:
                state.update(status='STOPPED_AFTER_FAILURE',end_utc=stamp());write(target,state)
                raise SystemExit(result.returncode)
    state.update(status='COMPLETE',end_utc=stamp());write(target,state)


if __name__=='__main__':main()
