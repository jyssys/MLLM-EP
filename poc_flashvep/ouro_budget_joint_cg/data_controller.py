"""Resume qualified data queues. Do not interpret model risk as deadline safety."""
import fcntl
import json
import subprocess
import time
from .common import ROOT,REPO,digest

def wait_plan(name):
    plan=json.loads((ROOT/name).read_text())
    jobs=plan.get('jobs',[j for g in plan.get('groups',[]) for j in g['jobs']])
    while True:
        failures=[j['id'] for j in jobs if (ROOT/'raw'/j['id']/'failure.json').exists()]
        assert not failures,('DATA_QUEUE_FAILURE_PRESERVED',failures)
        if all((ROOT/'raw'/j['id']/'completed.json').exists() for j in jobs):
            with (ROOT/'state/dispatcher.lock').open('a') as lease:
                try:
                    fcntl.flock(lease.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
                    return
                except BlockingIOError:
                    pass
        time.sleep(5)

def main():
    gate=json.loads((ROOT/'numerical_probe_gate.json').read_text())
    assert gate['integrity_pass'] and not gate['tolerance_failures']
    wait_plan('p1_observer_control_plan.json')
    for name in ('p1_fairness_plan.json','p2_service_plan.json','p2_controlled_service_plan.json'):
        plan=json.loads((ROOT/name).read_text())
        if all((ROOT/'raw'/j['id']/'completed.json').exists() for j in plan['jobs']):
            print('SKIP_COMPLETE_QUEUE',name,flush=True)
            continue
        subprocess.run(['/home/esjung/anaconda3/bin/python','-u','-m','poc_flashvep.ouro_budget_joint_cg.dispatch',
            '--manifest',str(ROOT/name),'--manage-burners'],cwd=REPO,check=True)
    subprocess.run(['/home/esjung/anaconda3/bin/python','-m','poc_flashvep.ouro_budget_joint_cg.analyze'],cwd=REPO,check=True)
    artifact=ROOT/'cost_model_artifacts/service_lookup_v1.json'
    if artifact.exists():
        freeze=json.loads((ROOT/'cost_model_artifacts/freeze_v1.json').read_text())
        assert digest(artifact)==freeze['sha256'],'Frozen model changed'
        print('SKIP_FROZEN_COST_MODEL',flush=True)
    else:
        subprocess.run(['/home/esjung/anaconda3/bin/python','-m','poc_flashvep.ouro_budget_joint_cg.cost_predictor','--fit'],cwd=REPO,check=True)
    print('DATA_QUEUE_COMPLETE_BUDGET_REVIEW_REQUIRED. No automatic capacity or quality claim.',flush=True)

if __name__=='__main__':
    main()
