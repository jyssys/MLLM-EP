"""Bounded residual-failure probes after the isolated transfer comparisons."""
import json
import os
import subprocess
import time
from .common import ROOT, REPO, PYTHON, environment, write, digest
from .resources import base
from .run import execute


def main():
    state=ROOT/'state/residual_followup.json'
    while True:
        path=ROOT/'state/quiet_experiments.json'
        if path.exists() and json.loads(path.read_text()).get('status') in ('COMPLETE','BLOCKED'):break
        write(state,dict(status='WAITING_FOR_QUIET_LANE',pid=os.getpid(),time=time.time()))
        time.sleep(10)
    # The quiet controller restores its owned burners after it publishes status.
    time.sleep(5)
    devices=list(range(8))
    outcomes=[]
    try:
        base.stop(devices)
        # The same readiness relaxation can apply to ordinary PD. This is a generic control.
        control_plan=[]
        for model in ('ouro','huginn'):
            for repeat in range(3):
                control_plan.append(dict(name=f'{model}_quiet_generic_queued_N8_cap8_r{repeat}',
                    model=model,variant='down_norm',arm='PD_G1',workload='diagnostic',dataset='W1',
                    count=8,cap=8,rate=0.,seed=2026103001,gpus=[4,5] if repeat!=1 else [6,7],
                    capacity=8,timeout=1800,diagnostic=True,quiet=True,method_label='PD_G1_queued',
                    independent_start=repeat,order=0,env={'B_NUMERICAL_QUEUED_COMPUTE_GENERIC':'1'}))
        write(ROOT/'configs/generic_queued_plan.json',control_plan)
        for job in control_plan:
            if base.processes():raise RuntimeError('Generic readiness control requires an empty quiet lane')
            write(state,dict(status='RUNNING_GENERIC_READINESS_CONTROL',job=job['name'],pid=os.getpid(),time=time.time()))
            outcomes.append(dict(name=job['name'],**execute(job)))
        cases=[]
        pairs=[('huginn_independent',85,'huginn_down_norm_qual_N200_seed2026109001_PD_G1_0',
                'huginn_down_norm_qual_N200_seed2026109001_PD_G1_1'),
               ('huginn_closed_loop',56,'huginn_down_norm_qual_N400_closed_loop_PD_G1_0',
                'huginn_down_norm_qual_N400_closed_loop_PD_G1_1')]
        for label,sample,left,right in pairs:
            paths=[ROOT/'raw'/name/'requests.jsonl' for name in (left,right)]
            rows=[next(r for line in p.read_text().splitlines()
                       if (r:=json.loads(line))['sample_id']==sample and not r.get('warmup')) for p in paths]
            a,b=rows
            position=next((j for j,(x,y) in enumerate(zip(a['output_ids'],b['output_ids'])) if x!=y),
                          min(len(a['output_ids']),len(b['output_ids'])))
            cases.append(dict(label=label,sample_id=sample,first_divergence=position,left=left,right=right,
                 left_row=a,right_row=b,input_equal=a['input_ids']==b['input_ids'],
                 history_equal=a['output_ids'][:position]==b['output_ids'][:position],
                 files={str(p.relative_to(ROOT)):digest(p) for p in paths}))
        manifest='configs/new_seed_failure_manifest.json'
        write(ROOT/manifest,cases)
        jobs=[]
        for model,gpu in [('ouro',4),('huginn',5)]:
            jobs.append((gpu,[PYTHON,'-m','poc_flashvep.looped_pd_b_numerical_diagnosis.component_timing',
                         '--model',model,'--name',f'{model}_hot_component_timing_r0']))
        for label,sample,_,_ in pairs:
            jobs.append((5,[PYTHON,'-m','poc_flashvep.looped_pd_b_numerical_diagnosis.teacher_replay',
                '--case',label,'--sample',str(sample),'--failure-manifest',manifest,
                '--sizes','1,2,4,8','--variant','down_norm','--track-all','--kv-hash',
                '--name',f'teacher_{label}_sample{sample}_residual_v8']))
        for gpu,command in jobs:
            if any(r['gpu']==gpu for r in base.processes()):raise RuntimeError('Residual diagnostic device is occupied')
            name=command[command.index('--name')+1]
            write(state,dict(status='RUNNING',job=name,completed=outcomes,pid=os.getpid(),time=time.time()))
            with (ROOT/'logs'/f'{name}.log').open('a') as log:
                try:
                    result=subprocess.run(command,cwd=REPO,env=environment([gpu]),stdout=log,stderr=subprocess.STDOUT,timeout=900)
                    outcomes.append(dict(name=name,returncode=result.returncode,command=command,gpu=gpu))
                except subprocess.TimeoutExpired:
                    outcomes.append(dict(name=name,status='TIMEOUT',command=command,gpu=gpu))
        write(state,dict(status='COMPLETE',results=outcomes,time=time.time(),
                       scope='Residual numerical diagnosis and hot operator timings. No serving-capacity campaign.'))
    finally:
        base.restore(devices)


if __name__=='__main__':main()
