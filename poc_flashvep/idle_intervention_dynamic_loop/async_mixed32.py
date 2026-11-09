"""Qualify actual negotiated cuts with complete state and transfer hashes."""
from .deferred_dynamic import *


def main():
    prepare();jobs=[]
    for model in ('ouro','huginn'):
        cfg=cell(model,32)
        cfg.update(name=f'n32_{model}_async_wire_actual_mixed_state32_v2',
            deferred_prompt_state_probe=False,async_prompt_state_probe=True,async_verify_handoff=True,
            minimum_P_pressure=2,verify_transfer=False,state_probe=True,
            qualification_ignore_observer_tail_latch=True,
            observer_scope='Full-byte observation changes timing. The latch exemption is restricted to state qualification. Actual instantaneous budgets and native state guards remain enabled. This run cannot establish Decode safety or performance.',
            phase='ACTUAL_BOUNDARY_NEGOTIATION_FULL_STATE_AND_WIRE_NOT_CLEAN_SLO')
        save(cfg);jobs.append(cfg['name'])
    p=ROOT/'state/queue.json';queue=json.loads(p.read_text())
    active=json.loads((ROOT/'state/active.json').read_text())['name'];i=queue.index(active)+1
    write(p,queue[:i]+jobs+queue[i:])


if __name__=='__main__':main()
