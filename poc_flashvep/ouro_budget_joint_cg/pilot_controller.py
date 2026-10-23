"""Wait for frozen data, qualify remaining frontiers, then run TRAIN pilots."""
import fcntl
import json
import subprocess
import time
from .common import ROOT,REPO,digest,write
from .data_controller import wait_plan

def run(module,*args):
    subprocess.run(['/home/esjung/anaconda3/bin/python','-u','-m',f'poc_flashvep.ouro_budget_joint_cg.{module}',*args],cwd=REPO,check=True)

def main():
    lock=(ROOT/'state/pilot_controller.lock').open('a')
    fcntl.flock(lock.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
    while not (ROOT/'cost_model_artifacts/freeze_v1.json').exists():
        time.sleep(5)
    for name in ('p2_service_plan.json','p2_controlled_service_plan.json'):
        wait_plan(name)
    artifact=ROOT/'cost_model_artifacts/service_lookup_v1.json'
    freeze=json.loads((ROOT/'cost_model_artifacts/freeze_v1.json').read_text())
    assert digest(artifact)==freeze['sha256']
    run('cost_interaction')
    run('dispatch','--manifest',str(ROOT/'p1_dynamic_frontier_plan.json'),'--manage-burners')
    jobs=json.loads((ROOT/'p1_dynamic_frontier_plan.json').read_text())['jobs']
    failures=[]
    for job in jobs:
        data=json.loads((ROOT/'raw'/job['id']/'completed.json').read_text())['summary']
        if not data['integrity_pass'] or not data['numerical_tolerance_pass']:
            failures.append(job['id'])
    write(ROOT/'dynamic_frontier_gate.json',dict(integrity_pass=not failures,failed_jobs=failures,
        scope='Pretrained isolated legal frontier cycle. Not batched full-answer quality.'))
    assert not failures,('BLOCKED_DYNAMIC_FRONTIER',failures)
    if not (ROOT/'p3_budget_pilot_plan.json').exists():
        run('pilot_plan')
    run('dispatch','--manifest',str(ROOT/'p3_budget_pilot_plan.json'),'--manage-burners')
    run('analyze')
    print('TRAIN_BUDGET_PILOT_COMPLETE_REVIEW_REQUIRED. No automatic final test or capacity claim.',flush=True)

if __name__=='__main__':
    main()
