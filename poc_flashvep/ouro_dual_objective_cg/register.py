"""Freeze the scope and repeated four-device interference controls."""
import hashlib
import itertools
import json
import subprocess
import time
from .common import ROOT, PRIOR, PARENT, SPEC, MODEL, FORK, GPUS, UUIDS, write, digest

def main():
    assert not (ROOT/'registration.json').exists(),'Keep the frozen registration'
    diff=subprocess.check_output(['git','diff','--binary'],cwd=FORK)
    old=json.loads((PRIOR/'model_source_locks.json').read_text())
    assert hashlib.sha256(diff).hexdigest()==old['diff_sha256']
    pin=subprocess.check_output(['git','rev-parse','HEAD'],cwd=FORK,text=True).strip()
    assert pin==old['pin']
    write(ROOT/'model_source_locks.json',dict(old,scope='Reuse the historical pinned runtime without edits'))
    write(ROOT/'gpu_uuid_map.json',{g:UUIDS[g] for g in GPUS})
    write(ROOT/'registration.json',dict(wall_s=time.time(),spec_sha256=digest(SPEC),parent=PARENT,
        GPUs=GPUS,forbidden_GPUs=[0,1,2,3],model_revision=MODEL.name,precision='BF16',R=4,
        drift_review_percent=3,clean_max_workers_until_review=1,
        C=[64,128,256,512,768,1024,1536,2048],G=[1,2,3,4],budgets_ms=[20,40,80,120,240],
        objective_families=[f'O{i}' for i in range(14)],lambda_grid=[0,.1,.25,.5,1,2,4,8],
        fresh_distinct_C_screen_fraction=.05,minimum_distinct_heldout_strata=2,
        incremental_goodput_screen_fraction=.05,paired_interval_must_exclude_zero=True,
        TTFT_p95_allowed_regression_fraction=.05,quality_loss_GSM8K_pp=1.,quality_loss_MATH_pp=2.,
        quality_pilot_does_not_prove_equivalence=True,capacity_min_offered_per_arm_rate=1000,
        capacity_min_starts=3,capacity_attainment=.90,deadline_guarantee=False,
        paused_native_vllm_loop_PP=True,preserved_staged_sha256='7018be31bd8a517574711055a72afbd6ae90737e4a585578c98ec255dbf7e1d9',
        background='GPUs0–3 remain untouched. Their pre-existing burners are exogenous shared-host load.'))
    groups=[]
    for repeat in range(2):
        for count in (1,2,4):
            for block,devices in enumerate([GPUS[i:i+count] for i in range(0,4,count)]):
                barrier=f'dual_p0_W{count}_S{repeat}_B{block}'
                jobs=[dict(id=f'dual_p0_W{count}_S{repeat}_gpu{g}_v1',kind='calibration',gpu=g,
                    workers=count,repeat=repeat,barrier=barrier,shapes=[[2048,4],[8192,8]],
                    repetitions=3,blocks=18432,budget=2048,timeout_s=2400) for g in devices]
                groups.append(dict(jobs=jobs,barrier=barrier))
    write(ROOT/'p0_contention_plan.json',dict(gpus=GPUS,groups=groups,
        scope='Physical4–7 reserved for 1/2/4-worker controls. Existing0–3 processes remain untouched.'))
    probes=[dict(id=f'dual_p1_frontier_gpu{g}_v1',kind='numerical',gpu=g,
        lengths=[128,512] if g%2==0 else [1536,2048],
        modes=['atomic','grouped','grouped3','scripted','cycled'],graphs=True,streams=True,timeout_s=2400) for g in GPUS]
    write(ROOT/'p1_numerical_plan.json',dict(gpus=GPUS,jobs=probes,
        scope='Pretrained hidden/logit/KV checks, legal C/G cycles, and Full-R4 accounting. Not benchmark accuracy.'))
    write(ROOT/'research_state.json',dict(status='RUNNING',phase='A_COMPLETE_B_PENDING',
        next='Run contention and numerical probes while constructing offline objectives.',
        task='Dual-objective Chunk–Loop scheduling',native_vllm_PP='PAUSED'))
    print('Registered',sum(len(g['jobs']) for g in groups),'calibration jobs and',len(probes),'numerical jobs')

if __name__=='__main__': main()
