"""Do not restrict the Token comparator to small chunks."""
from .selection import *


def main():
    prepare();jobs=[]
    for model in ('ouro','huginn'):
        T=4096 if model=='ouro' else 2048
        cells=([(3072,width,quantum) for width in (512,1024,3072) for quantum in (1,4)]
            if model=='ouro' else [(512,512,q) for q in (1,4,8,16,32)]
                +[(1024,1024,q) for q in (1,4,8,32)])
        for suffix,width,quantum in cells:
            cfg=token_cost(model,width,quantum)
            cfg.update(name=f'ptw_{model}_suffix{suffix}_tile{width}_quantum{quantum}_serial8',
                native_frontier=[0,T-suffix],deep_graph_budget=64,P_prefill_graph_shapes=64)
            save(cfg);jobs.append(cfg['name'])
        for width in sorted({512,1024,T//2}):
            for quantum in (1,2):
                cfg=extra_cost(model,width,2,quantum)
                cfg.update(name=f'pw_{model}_tile{width}_loops2_quantum{quantum}_serial8',
                    deep_graph_budget=64,P_prefill_graph_shapes=64)
                save(cfg);jobs.append(cfg['name'])
    p=ROOT/'state/queue.json';write(p,json.loads(p.read_text())+jobs)


if __name__=='__main__':main()
