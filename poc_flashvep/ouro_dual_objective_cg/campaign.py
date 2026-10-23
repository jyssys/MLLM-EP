"""Resume-safe stage handoff. Never restart completed or failed attempts."""
import argparse
import fcntl
import json
import os
import subprocess
import time
from .common import ROOT,REPO,write

def wait_for(plan):
    jobs=json.loads(plan.read_text())['jobs']
    while True:
        failures=[j['id'] for j in jobs if (ROOT/'raw'/j['id']/'failure.json').exists()]
        assert not failures,('Review failures before promotion',failures)
        if all((ROOT/'raw'/j['id']/'completed.json').exists() for j in jobs):return
        time.sleep(5)

def main():
    p=argparse.ArgumentParser();p.add_argument('--validation',action='store_true');a=p.parse_args()
    (ROOT/'state').mkdir(parents=True,exist_ok=True)
    handle=(ROOT/'state'/'coordinator.lock').open('a')
    fcntl.flock(handle,fcntl.LOCK_EX|fcntl.LOCK_NB)
    write(ROOT/'state'/'coordinator.json',dict(pid=os.getpid(),start_wall_s=time.time(),stage='WAIT_ADAPTIVE_SMOKE'))
    if a.validation:
        wait_for(ROOT/'p1_adaptive_smoke_retry_v2_plan.json')
        write(ROOT/'research_state.json',dict(status='RUNNING',phase='B_LIVE_VALIDATION',
            next='Complete the 22-arm matched objective screen, then lock FINAL policy using VALIDATION only.',
            native_vllm_loop_PP='PAUSED',GPUs=[4,5,6,7]))
        code=subprocess.call(['/home/esjung/anaconda3/bin/python','-u','-m','poc_flashvep.ouro_dual_objective_cg.dispatch',
            '--manifest',str(ROOT/'p2_objective_validation_plan.json'),'--manage-burners'],cwd=REPO)
        write(ROOT/'state'/'coordinator.json',dict(pid=os.getpid(),end_wall_s=time.time(),
            stage='VALIDATION_COMPLETE' if code==0 else 'VALIDATION_REVIEW_REQUIRED',exit_code=code))
        assert code==0,'Preserve failures and review a new attempt ID'

if __name__=='__main__':main()
