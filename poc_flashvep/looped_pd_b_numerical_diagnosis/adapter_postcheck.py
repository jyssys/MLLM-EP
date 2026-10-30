"""Small byte, order, geometry and HF-triage checks after expanded adapter gates."""
import json
import os
import subprocess
import time
from .common import ROOT,REPO,PYTHON,environment,write
from .resources import base
from .run import execute


def main():
    state=ROOT/'state/huginn_adapter_postcheck.json'
    while True:
        p=ROOT/'state/huginn_adapter_qualification.json'
        value=json.loads(p.read_text()) if p.exists() else {}
        if value.get('status') in ('FINITE_QUALIFICATION_PASS','NO_GO_CORRECTNESS','NO_GO_OPERATOR_REPLAY'):break
        write(state,dict(status='WAITING_FOR_EXPANDED_GATES',pid=os.getpid(),time=time.time()))
        time.sleep(15)
    if value['status']!='FINITE_QUALIFICATION_PASS':
        write(state,dict(status='SKIPPED_FAILED_DEPENDENCY',qualification=value['status'],time=time.time()));return
    time.sleep(5)
    devices=[4,5];results=[]
    try:
        base.stop(devices)
        common=dict(model='huginn',variant='down_norm_adapter',workload='diagnostic',seed=2026103001,
                    gpus=devices,timeout=1800,diagnostic=True)
        jobs=[dict(common,name='huginn_adapter_prompt_KV_B_0',arm='B',dataset='W1',count=8,cap=8,rate=0.,
                   capacity=8,env={'ABC_VERIFY_KV':'1'})]
        for index,arm in enumerate(('PD_G1','B')):
            jobs.append(dict(common,name=f'huginn_adapter_order8_capacity3_{arm}_{index}',arm=arm,
                dataset='W1_reversed8',count=8,cap=128,rate=0.,capacity=3,
                env={'B_NUMERICAL_PRESERVE_SOURCE_SEEDS':'1'}))
        jobs.append(dict(common,name='huginn_adapter_native_HF_triage8_Native_0',arm='Native',dataset='correctness',
             count=8,cap=32,rate=0.,capacity=1,seed=20261029,gpus=[4],env={}))
        write(ROOT/'configs/huginn_adapter_postcheck_plan.json',jobs)
        for job in jobs:
            write(state,dict(status='RUNNING',job=job['name'],completed=results,pid=os.getpid(),time=time.time()))
            results.append(dict(name=job['name'],**execute(job)))
        name='teacher_huginn_adapter_geometry16_v10'
        command=[PYTHON,'-m','poc_flashvep.looped_pd_b_numerical_diagnosis.teacher_replay',
            '--case','huginn_closed_loop','--sample','56','--failure-manifest','configs/new_seed_failure_manifest.json',
            '--sizes','1,2,3,4,8,16','--variant','down_norm_adapter','--track-all','--kv-hash','--name',name]
        with (ROOT/'logs'/f'{name}.log').open('a') as log:
            result=subprocess.run(command,cwd=REPO,env=environment([4]),stdout=log,stderr=subprocess.STDOUT,timeout=900)
        results.append(dict(name=name,command=command,returncode=result.returncode))
        write(state,dict(status='COMPLETE',results=results,time=time.time()))
    except BaseException as error:
        write(state,dict(status='BLOCKED_RUNTIME',results=results,error=repr(error),time=time.time()))
        raise
    finally:
        base.restore(devices)


if __name__=='__main__':main()
