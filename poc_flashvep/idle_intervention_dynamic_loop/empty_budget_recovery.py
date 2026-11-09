"""Use the same declared empty-D cap for both Token and Loop admission."""
import copy
from .predictive32 import prototype
from .common import *


def main():
    initialize();active=json.loads((ROOT/'state/active.json').read_text())['name'];jobs=[]
    for model in ('ouro','huginn'):
        for count in (8,32):
            cfg=prototype(model,'tile',count=count)
            cfg.update(name=f'r6_{model}_actual_mixed_state{count}',empty_admission_uses_current_cap=True,
                state_probe=True,verify_transfer=False,async_prompt_state_probe=True,async_verify_handoff=True,
                state_probe_measured_only=True,state_hash_defer_until_close=True,state_pinned_slots=count+2,
                qualification_ignore_observer_tail_latch=True,adaptive_decode_reserve=False,
                timeout=900,startup_timeout=300,phase='CURRENT_EMPTY_CAP_ACTUAL_MIXED_FULL_STATE',
                budget_scope='No active Decode has an observed outstanding ITL deadline. The existing 200 ms cap is a policy limit, not a measured idle gap or a future-arrival guarantee. Commit and execution recheck current Decode state.')
            save(cfg)
            if count==8:jobs.append(cfg['name'])
        for arm in ('token','loop','tile'):
            cfg=prototype(model,arm,count=32)
            cfg.update(name=f'r6_{model}_{arm}_selection32',empty_admission_uses_current_cap=True,
                phase='SHARED_EMPTY_CAP_POLICY_SELECTION_NOT_CONFIRMATION')
            save(cfg)
    path=ROOT/'state/queue.json';queue=json.loads(path.read_text());index=queue.index(active)+1
    write(path,queue[:index]+jobs+queue[index:]);print(json.dumps(jobs))


if __name__=='__main__':main()
