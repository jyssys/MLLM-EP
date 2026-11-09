"""Check every transferred byte without blocking progress for CPU SHA work."""
from .deferred_dynamic import *
from .cells import intervention


def main():
    prepare();jobs=[]
    for model in ('ouro','huginn'):
        for dynamic in (False,True):
            cfg=cell(model) if dynamic else intervention(model,4096 if model=='ouro' else 2048)
            cfg.update(name=f'n8_{model}_async_wire_{"dynamic" if dynamic else "static"}_state8',
                deferred_prompt_state_probe=False,async_prompt_state_probe=True,async_verify_handoff=True,
                verify_transfer=False,state_probe=True,dedicated_handoff_copy_stream=True,
                minimum_P_pressure=1,deep_graph_budget=64,P_prefill_graph_shapes=64,
                phase='ASYNC_FULL_BYTE_HANDOFF_AND_STATE_QUALIFICATION_NOT_PERFORMANCE')
            save(cfg);jobs.append(cfg['name'])
    p=ROOT/'state/queue.json';queue=json.loads(p.read_text())
    active=json.loads((ROOT/'state/active.json').read_text())['name'];i=queue.index(active)+1
    write(p,queue[:i]+jobs+queue[i:])


if __name__=='__main__':main()
