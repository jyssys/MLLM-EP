"""Automatic safe stage handoff after the frozen validation queue."""
import fcntl
import json
import os
import subprocess
import time
from .common import ROOT,REPO,write

PYTHON='/home/esjung/anaconda3/bin/python'

def command(module,*arguments):
    code=subprocess.call([PYTHON,'-u','-m',f'poc_flashvep.ouro_dual_objective_cg.{module}',*arguments],cwd=REPO,
        env=dict(os.environ,OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1'))
    assert code==0,(module,code,'Preserve the failed attempt and review before continuing')

def main():
    (ROOT/'state').mkdir(parents=True,exist_ok=True)
    own=(ROOT/'state'/'followup.lock').open('a')
    fcntl.flock(own,fcntl.LOCK_EX|fcntl.LOCK_NB)
    lease=(ROOT/'state'/'coordinator.lock').open('a')
    fcntl.flock(lease,fcntl.LOCK_EX)  # The prior coordinator must finish restoration first.
    validation=json.loads((ROOT/'p2_objective_validation_plan.json').read_text())
    assert all((ROOT/'raw'/j['id']/'completed.json').exists() for j in validation['jobs'])
    write(ROOT/'state'/'followup.json',dict(pid=os.getpid(),stage='FRESH_TRAIN_FIXED_TUNING',wall_s=time.time()))
    command('dispatch','--manifest',str(ROOT/'p2_fixed_tuning_plan.json'),'--manage-burners')
    command('analyze_serving','--require-validation-complete')
    command('select_final')
    command('final_plan')
    write(ROOT/'state'/'followup.json',dict(pid=os.getpid(),stage='LOCKED_FINAL_WORKLOAD_CHARACTERIZATION',wall_s=time.time()))
    command('dispatch','--manifest',str(ROOT/'p3_final_workload_plan.json'),'--manage-burners')
    command('analyze_serving','--require-validation-complete')
    write(ROOT/'state'/'followup.json',dict(pid=os.getpid(),stage='FROZEN_PUBLIC_QUALITY_PILOT',wall_s=time.time()))
    command('dispatch','--manifest',str(ROOT/'p4_quality_plan.json'),'--manage-burners')
    command('grade_quality')
    write(ROOT/'research_state.json',dict(status='MEASUREMENTS_COMPLETE_REVIEW_REQUIRED',
        phase='REVIEW_FINAL_QUALITY_AND_CAPACITY_GATE',native_vllm_loop_PP='PAUSED',
        next='Review FINAL measured contrasts and accuracy. Register qualified-rate tests only if gates pass. Then complete reports and scoped commit/push.'))
    write(ROOT/'state'/'followup.json',dict(pid=os.getpid(),stage='FINAL_AND_QUALITY_COMPLETE_REVIEW_GATE',wall_s=time.time()))

if __name__=='__main__':main()
