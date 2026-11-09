"""Add bounded context controls after the frozen primary confirmation plan."""
from .common import *
from .policy import configured


def main():
    assert (ROOT/'heldout_plan.json').exists()
    jobs=[]
    for model in ('ouro','huginn'):
        for load in ('low','medium'):
            cfg=configured(model,f'context_{model}_native_{load}16','native',16,2026110914,load)
            cfg.update(phase='SECONDARY_NATIVE_LOAD_CONTEXT_NOT_METHOD_CONFIRMATION',
                timeout=300,startup_timeout=180)
            save(cfg);jobs.append(cfg['name'])
    for model in ('ouro','huginn'):
        for arm in ('static_token','static_loop'):
            cfg=configured(model,f'context_{model}_{arm}_selection32',arm,32,2026110801)
            cfg.update(phase='SECONDARY_FIXED_CUT_SELECTION_NOT_HELD_OUT_CONFIRMATION',
                timeout=300,startup_timeout=180,
                comparison_reference=f'r1_{model}_native_selection32',
                empty_admission_uses_current_cap=True)
            save(cfg);jobs.append(cfg['name'])
    path=ROOT/'state/queue.json';queue=json.loads(path.read_text())
    assert not any(name in queue for name in jobs)
    write(path,queue+jobs)
    print(json.dumps(jobs))


if __name__=='__main__':main()
