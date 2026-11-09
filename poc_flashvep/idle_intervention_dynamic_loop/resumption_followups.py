"""Freeze paired policy tests after the completed large-quantum calibration."""
from .predictive32 import prototype
from .common import *


def main():
    assert RESUMPTION is not None
    initialize();jobs=[]
    dependencies=json.loads((ROOT/'state/resumption_gate_dependencies.json').read_text())
    for model in ('ouro','huginn'):
        for count in (8,32):
            cfg=prototype(model,'token',count=count)
            cfg.update(name=f'r1_{model}_actual_token_state{count}',state_probe=True,
                verify_transfer=False,async_prompt_state_probe=True,async_verify_handoff=True,
                deferred_prompt_state_probe=False,qualification_ignore_observer_tail_latch=True,
                adaptive_decode_reserve=False,timeout=900,startup_timeout=300,
                phase='ACTUAL_ONLINE_TOKEN_FULL_BYTE_DIAGNOSTIC_NOT_PERFORMANCE',
                resumption_record='state/resumption_20261109.json')
            save(cfg);jobs.append(cfg['name'])
            dependencies[cfg['name']]=[f'r1_{model}_pinned_static_state8'] if count==8 else [f'r1_{model}_actual_token_state8']
        for arm in ('native','zero','rejection','loop','tile','token'):
            cfg=prototype(model,arm,count=32)
            cfg.update(name=f'r1_{model}_{arm}_selection32',timeout=600,startup_timeout=300,
                phase='PAIRED_POLICY_SELECTION_NOT_HELD_OUT_CONFIRMATION',
                resumption_record='state/resumption_20261109.json')
            save(cfg);jobs.append(cfg['name'])
    write(ROOT/'state/resumption_gate_dependencies.json',dependencies)
    path=ROOT/'state/queue.json';queue=json.loads(path.read_text())
    write(path,queue+[name for name in jobs if name not in queue])
    print(json.dumps(jobs,indent=2))


if __name__=='__main__':main()
