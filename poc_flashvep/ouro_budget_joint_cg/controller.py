"""Wait without using GPUs, then execute only the next qualified queue."""
import fcntl
import json
import subprocess
import time
from .common import ROOT,REPO

def main():
    plan=json.loads((ROOT/'p0_contention_plan.json').read_text())
    jobs=[j for group in plan['groups'] for j in group['jobs']]
    while True:
        failures=[j['id'] for j in jobs if (ROOT/'raw'/j['id']/'failure.json').exists()]
        assert not failures,('CALIBRATION_FAILURE',failures)
        complete=all((ROOT/'raw'/j['id']/'completed.json').exists() for j in jobs)
        if complete:
            with (ROOT/'state/dispatcher.lock').open('a') as lease:
                try:
                    fcntl.flock(lease.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
                    break
                except BlockingIOError:
                    pass
        time.sleep(5)
    subprocess.run(['/home/esjung/anaconda3/bin/python','-m','poc_flashvep.ouro_budget_joint_cg.contention'],cwd=REPO,check=True)
    subprocess.run(['/home/esjung/anaconda3/bin/python','-m','poc_flashvep.ouro_budget_joint_cg.dispatch','--manifest',
        str(ROOT/'p1_numerical_plan.json'),'--manage-burners'],cwd=REPO,check=True)
    print('NUMERICAL_QUEUE_COMPLETE_REVIEW_REQUIRED. No automatic broad serving or model GO.',flush=True)

if __name__=='__main__':
    main()
