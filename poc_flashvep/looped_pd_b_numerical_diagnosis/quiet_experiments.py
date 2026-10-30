"""Small, isolated root-cause cohorts after all numerical qualification finishes."""
import json
import os
import subprocess
import time
from .common import ROOT,setup,write
from .run import execute
from .resources import base


def main():
    setup()
    state=ROOT/'state/quiet_experiments.json'
    terminal={'FINITE_QUALIFICATION_PASS','NO_GO_CORRECTNESS','BLOCKED_RUNTIME'}
    while True:
        paths=[ROOT/'state'/f'{model}_down_norm_qualification.json' for model in ('ouro','huginn')]
        qualification_ready=all(p.exists() and json.loads(p.read_text()).get('status') in terminal for p in paths)
        h400=ROOT/'state/huginn_400_qualification.json'
        qualification_ready=qualification_ready and h400.exists() and json.loads(h400.read_text()).get('status') in terminal
        extras=[ROOT/'raw'/name/'status.json' for name in ('ouro_old_canonical_seed7001_N400_B_2',
                    'ouro_old_canonical_sample233_capture_B_1','huginn_new_native_HF_triage8_Native_0')]
        other_ready=all(p.exists() and json.loads(p.read_text()).get('status') in ('COMPLETE','FAILED') for p in extras)
        if qualification_ready and other_ready:break
        write(state,dict(status='WAITING_FOR_NUMERICAL_JOBS',pid=os.getpid(),time=time.time()))
        time.sleep(10)
    # Closing numerical processes can retain CUDA contexts briefly after status publication.
    time.sleep(3)
    try:
        base.stop(list(range(8)))
        if base.processes():raise RuntimeError('Quiet lane is not empty. Leave unrelated work untouched.')
        plan=[]
        for model in ('ouro','huginn'):
            for repeat in range(3):
                for position,method in enumerate(('PD_G1','B','B_queued') if repeat%2==0 else ('B_queued','B','PD_G1')):
                    name=f'{model}_quiet_down_norm_N8_cap8_{method}_r{repeat}'
                    plan.append(dict(name=name,model=model,variant='down_norm',arm='PD_G1' if method=='PD_G1' else 'B',
                        workload='diagnostic',dataset='W1',count=8,cap=8,rate=0.,seed=2026103001,
                        gpus=[4,5] if repeat!=1 else [6,7],capacity=8,timeout=1800,diagnostic=True,quiet=True,
                        method_label=method,independent_start=repeat,order=position,
                        env={'B_NUMERICAL_QUEUED_COMPUTE':'1'} if method=='B_queued' else {}))
        # These two starts measure instrumentation costs separately. They are not pooled with clean latency.
        for model in ('ouro','huginn'):
            for method in ('PD_G1','B','B_queued'):
                plan.append(dict(name=f'{model}_quiet_host_probe_{method}',model=model,variant='down_norm',
                    arm='PD_G1' if method=='PD_G1' else 'B',workload='diagnostic',dataset='W1',
                    count=8,cap=8,rate=0.,seed=2026103001,gpus=[4,5],capacity=8,timeout=1800,
                    diagnostic=True,quiet=True,method_label=method,
                    env=dict(B_NUMERICAL_TRANSFER_PROBE='1',**({'B_NUMERICAL_QUEUED_COMPUTE':'1'} if method=='B_queued' else {}))))
        for model in ('ouro','huginn'):
            for method in ('PD_G1','B'):
                plan.append(dict(name=f'{model}_quiet_nsys_{method}',model=model,variant='down_norm',arm=method,
                    workload='diagnostic',dataset='W1',count=8,cap=8,rate=0.,seed=2026103001,gpus=[4,5],
                    capacity=8,timeout=1800,diagnostic=True,quiet=True,method_label=method,nsys=True,
                    env={'B_NUMERICAL_TRANSFER_PROBE':'1'}))
        write(ROOT/'configs/quiet_plan.json',plan)
        results=[]
        for job in plan:
            # No models, burners, or other work may overlap these finite comparisons.
            active=base.processes()
            if active:raise RuntimeError('A GPU remains occupied before the next isolated job: '+str(active))
            write(state,dict(status='RUNNING',job=job['name'],completed=results,pid=os.getpid(),time=time.time()))
            value=execute(job)
            results.append(dict(name=job['name'],**value))
            if value.get('status')!='COMPLETE':
                # Preserve failures. Continue independent diagnostic methods.
                print('DIAGNOSTIC_FAILURE',job['name'],flush=True)
        write(state,dict(status='COMPLETE',results=results,time=time.time()))
    except BaseException as error:
        write(state,dict(status='BLOCKED',error=repr(error),time=time.time()))
        raise
    finally:
        base.restore(list(range(8)))


if __name__=='__main__':main()
