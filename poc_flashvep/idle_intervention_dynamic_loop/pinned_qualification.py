"""Validate the reusable observer buffers before another mixed state cohort."""
from .async_wire_qualification import *


def main():
    prepare();jobs=[]
    for model in ('ouro','huginn'):
        for dynamic in (False,True):
            cfg=cell(model) if dynamic else intervention(model,4096 if model=='ouro' else 2048)
            cfg.update(name=f'n8_{model}_pinned_wire_{"dynamic" if dynamic else "static"}_state8',
                deferred_prompt_state_probe=False,async_prompt_state_probe=True,async_verify_handoff=True,
                verify_transfer=False,state_probe=True,dedicated_handoff_copy_stream=True,
                qualification_ignore_observer_tail_latch=True,minimum_P_pressure=2,
                phase='PINNED_BUFFER_FULL_BYTE_QUALIFICATION_NOT_PERFORMANCE')
            save(cfg);jobs.append(cfg['name'])
    p=ROOT/'state/queue.json';queue=json.loads(p.read_text())
    active=json.loads((ROOT/'state/active.json').read_text())['name'];i=queue.index(active)+1
    write(p,queue[:i]+jobs+queue[i:])


if __name__=='__main__':main()
