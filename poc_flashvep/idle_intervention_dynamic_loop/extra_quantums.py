"""Give Token and Loop execution the same measured grouping options."""
from .selection import *


def main():
    prepare();jobs=[]
    for model in ('ouro','huginn'):
        for width in (32,128,256):
            for quantum in ((4,) if model=='ouro' else (4,8,16,32)):
                cfg=token_cost(model,width,quantum)
                cfg['name']=f'ptg_{model}_suffix512_tile{width}_quantum{quantum}_serial8'
                save(cfg);jobs.append(cfg['name'])
        if model=='huginn':
            for width in (128,256,2048):
                cfg=extra_cost(model,width,8,8)
                cfg['name']=f'pg_huginn_tile{width}_loops8_quantum8_serial8'
                save(cfg);jobs.append(cfg['name'])
        for token in (False,True):
            cfg=intervention(model,128,state=True)
            cfg.update(name=f'n8_{model}_{"token" if token else "loop"}_grouped_state8',
                D_loop_quantum=cfg['depth'] if token else 3 if model=='ouro' else 8,
                dedicated_handoff_copy_stream=True,deep_graph_budget=64,P_prefill_graph_shapes=64)
            if token:
                cfg.update(intervention_method='static_token',native_frontier=[0,cfg['length']//2])
                cfg.pop('native_partial_cut',None)
            else:cfg['native_partial_cut']=1 if model=='ouro' else 24
            save(cfg);jobs.append(cfg['name'])
    p=ROOT/'state/queue.json';queue=json.loads(p.read_text())
    # Execute these before policy confirmation, but do not interrupt an active job.
    write(p,queue+jobs)


if __name__=='__main__':main()
