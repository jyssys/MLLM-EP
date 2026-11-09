"""The asynchronous probe must reproduce the original SHA256 records."""
from .deferred_qualification import *


def main():
    prepare();jobs=[]
    for model in ('ouro','huginn'):
        cfg=configuration(model,f'n8_{model}_async_native_state8',8,state=True,method='zero')
        cfg.update(async_prompt_state_probe=True,verify_transfer=False,
            phase='ASYNC_HASH_EQUIVALENCE_NOT_PERFORMANCE',
            D_tile_tokens=cfg['length'],D_loop_quantum=1,minimum_quantum_ms=0.,empty_decode_budget_ms=200.)
        save(cfg);jobs.append(cfg['name'])
    p=ROOT/'state/queue.json';queue=json.loads(p.read_text())
    active=json.loads((ROOT/'state/active.json').read_text())['name'];i=queue.index(active)+1
    write(p,queue[:i]+jobs+queue[i:])


if __name__=='__main__':main()
