"""Predeclare Native A/A repeats and grouped-tile state checks."""
from .calibration import *


def main():
    prepare();jobs=[]
    for model in ('ouro','huginn'):
        for repeat in (2,3):
            cfg=configuration(model,f'aa_{model}_same_trace24_r{repeat}')
            cfg['phase']='NATIVE_A_A_SAME_TRACE_INDEPENDENT_START'
            save(cfg);jobs.append(cfg['name'])
        cfg=intervention(model,128)
        cfg.update(name=f'q_{model}_tile128_loops2_state8',native_partial_cut=cfg['depth']-2,D_loop_quantum=2)
        save(cfg);jobs.append(cfg['name'])
    q=json.loads((ROOT/'state/queue.json').read_text())
    at=next(i for i,name in enumerate(q) if name.startswith('p_'))
    write(ROOT/'state/queue.json',q[:at]+jobs+q[at:])


if __name__=='__main__':main()
