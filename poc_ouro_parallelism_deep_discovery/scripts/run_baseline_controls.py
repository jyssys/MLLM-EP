"""Finite, resumable generic-baseline queue. Resource waits are not research time."""
import argparse
import datetime as dt
import subprocess
import time
from poc_ouro_parallelism_deep_discovery.contract import *

def stamp():return dt.datetime.now(dt.timezone.utc).isoformat()

def main():
    p=argparse.ArgumentParser();p.add_argument('--after',type=Path,required=True);args=p.parse_args()
    target=ROOT/'artifacts/baseline_control_queue.json';assert not target.exists()
    queue=[('control_async_base14_P1_v1','6','1.4B',1),
           ('control_async_base26_P1_v1','6','2.6B',1),
           ('control_async_base26_P2_v1','4,6','2.6B',2),
           ('control_async_base26_P4_v1','4,5,6,7','2.6B',4)]
    state=dict(pid=os.getpid(),start_utc=stamp(),jobs=[],status='WAITING_FOR_OWN_PREREQUISITE',
        prerequisite=str(args.after),waiting_is_substantive_research=False,
        purpose='Existing native option baseline control. Not a novel architecture or automatic ten-hour campaign.')
    write(target,state)
    deadline=time.monotonic()+1200
    while not args.after.exists():
        if time.monotonic()>deadline:raise TimeoutError('Owned prerequisite did not finish within1200 seconds.')
        time.sleep(1)
    assert read(args.after).get('success'),args.after
    state['prerequisite_finished_utc']=stamp()
    for name,gpus,size,tp in queue:
        previous=ROOT/'outputs'/name/'guard_status.json'
        if previous.exists():
            assert read(previous).get('success'),previous
            state['jobs'].append(dict(name=name,status='SKIP_ALREADY_SUCCESSFUL'))
            write(target,state);continue
        assert not (ROOT/'outputs'/name).exists(),name
        command=[REFERENCE_PYTHON,'-m','poc_ouro_parallelism_deep_discovery.scripts.gpu_session',
            '--gpus',gpus,'--output','outputs/'+name,'--timeout','3600','--',PYTHON,
            '-m','poc_ouro_parallelism_deep_discovery.scripts.native_baseline',
            '--size',size,'--tp',str(tp),'--R','4','--async-scheduling']
        record=dict(name=name,physical_gpus=list(map(int,gpus.split(','))),size=size,TP=tp,
            command=command,start_utc=stamp(),status='RUNNING')
        state['jobs'].append(record);state['status']='RUNNING';write(target,state)
        print('CONTROL_QUEUE_START',name,stamp(),flush=True)
        result=subprocess.run(command,cwd=PROJECT)
        record.update(exit_code=result.returncode,end_utc=stamp(),status='PASS' if result.returncode==0 else 'FAIL')
        write(target,state)
        if result.returncode:
            state['status']='STOPPED_AFTER_FAILURE';state['end_utc']=stamp();write(target,state)
            raise SystemExit(result.returncode)
    state.update(status='COMPLETE',end_utc=stamp());write(target,state)
    print('CONTROL_QUEUE_COMPLETE',stamp(),flush=True)

if __name__=='__main__':main()
