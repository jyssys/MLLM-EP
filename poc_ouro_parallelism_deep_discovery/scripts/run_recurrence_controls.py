"""Finite R1 hardware ablation. R1 is not quality-equivalent to fixed R4."""
import datetime as dt
import subprocess
import time
from poc_ouro_parallelism_deep_discovery.contract import *

def stamp():return dt.datetime.now(dt.timezone.utc).isoformat()

def wait_owned(path,timeout=3600):
    deadline=time.monotonic()+timeout
    while not path.exists():
        if time.monotonic()>deadline:raise TimeoutError(str(path))
        time.sleep(1)
    assert read(path).get('success'),path

def main():
    target=ROOT/'artifacts/recurrence_control_queue.json';assert not target.exists()
    state=dict(pid=os.getpid(),start_utc=stamp(),status='WAITING_FOR_OWN_PREREQUISITE',jobs=[],
        waiting_is_substantive_research=False,
        purpose='R1 counterfactual for generic operator and recurrent reuse effects. Not task quality or candidate architecture performance.')
    write(target,state)
    wait_owned(ROOT/'outputs/graph_intervals_base26_P2_v1/guard_status.json')
    queue=[('1.4B',1,'6'),('1.4B',2,'4,6'),('2.6B',1,'6'),('2.6B',2,'4,6'),
           ('1.4B',4,'4,5,6,7'),('2.6B',4,'4,5,6,7')]
    for size,tp,gpus in queue:
        if tp==4:
            for model in ['14','26']:
                wait_owned(ROOT/f'outputs/quality_native_async_base{model}_P1_v1/guard_status.json')
        short=size.replace('.','').replace('B','')
        name=f'control_asyncR1_base{short}_P{tp}_v1'
        assert not (ROOT/'outputs'/name).exists()
        command=[REFERENCE_PYTHON,'-m','poc_ouro_parallelism_deep_discovery.scripts.gpu_session',
            '--gpus',gpus,'--output','outputs/'+name,'--timeout','3600','--',PYTHON,
            '-m','poc_ouro_parallelism_deep_discovery.scripts.native_baseline',
            '--size',size,'--tp',str(tp),'--R','1','--async-scheduling']
        job=dict(name=name,physical_gpus=list(map(int,gpus.split(','))),command=command,
            start_utc=stamp(),status='RUNNING')
        state['jobs'].append(job);state['status']='RUNNING';write(target,state)
        print('R1_CONTROL_START',name,stamp(),flush=True)
        result=subprocess.run(command,cwd=PROJECT)
        job.update(status='PASS' if result.returncode==0 else 'FAIL',exit_code=result.returncode,end_utc=stamp())
        write(target,state)
        if result.returncode:
            state.update(status='STOPPED_AFTER_FAILURE',end_utc=stamp());write(target,state)
            raise SystemExit(result.returncode)
    state.update(status='COMPLETE',end_utc=stamp());write(target,state)

if __name__=='__main__':main()
