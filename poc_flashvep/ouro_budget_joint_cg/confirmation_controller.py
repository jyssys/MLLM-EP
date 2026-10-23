"""Wait for quiet confirmation, then run a frozen matched quality pilot."""
import fcntl
import json
import os
import subprocess
from .common import ROOT,REPO,FORK,PYTHON
from .data_controller import wait_plan

def run(module,*args):
    subprocess.run(['/home/esjung/anaconda3/bin/python','-u','-m',f'poc_flashvep.ouro_budget_joint_cg.{module}',*args],cwd=REPO,check=True)

def main():
    lock=(ROOT/'state/confirmation_controller.lock').open('a')
    fcntl.flock(lock.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
    wait_plan('p5_quiet_confirmation_plan.json')
    run('analyze');run('live_pairs');run('budget_analysis');run('figures')
    manifest=ROOT/'quality_manifest.json'
    if not manifest.exists():
        env=dict(os.environ,PYTHONPATH=str(FORK)+':'+str(REPO))
        subprocess.run([PYTHON,'-m','poc_flashvep.ouro_budget_joint_cg.quality','--budget','40'],cwd=REPO,env=env,check=True)
    assert json.loads(manifest.read_text())['budget_ms']==40
    run('dispatch','--manifest',str(ROOT/'p6_quality_plan.json'),'--manage-burners')
    run('grade_quality')
    print('QUIET_CONFIRMATION_AND_QUALITY_PILOT_COMPLETE_REVIEW_REQUIRED.',flush=True)

if __name__=='__main__':
    main()
