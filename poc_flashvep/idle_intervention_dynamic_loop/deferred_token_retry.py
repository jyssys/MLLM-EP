"""Qualify the actual-length startup fix in immutable namespaces."""
from .common import *


def main():
    initialize();jobs=[]
    for model in ('ouro','huginn'):
        old=f'n8_{model}_deferred_token_state8'
        cfg=json.loads((ROOT/'configs'/(old+'.json')).read_text())
        cfg.update(name=old+'_v2',async_prompt_state_probe=True,async_verify_handoff=True,
            verify_transfer=False,state_probe=True,
            phase='DEFERRED_ARMING_STARTUP_AND_EXACT_STATE_QUALIFICATION_NOT_SLO',
            retry_reason='Unarmed startup uses each request actual prompt length, including short graph warm-up inputs.')
        save(cfg);jobs.append(cfg['name'])
    p=ROOT/'state/queue.json';queue=json.loads(p.read_text())
    active=json.loads((ROOT/'state/active.json').read_text())
    i=queue.index(active['name'])+1 if active.get('name') else 0
    write(p,queue[:i]+jobs+queue[i:])


if __name__=='__main__':main()
