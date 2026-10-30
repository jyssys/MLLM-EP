"""A limited long-prompt fixed-cut control. It is not a tuned optimum search."""
import json
from .common import ROOT,MODEL_PINS,write


def main():
    plans=[]
    for model in ('ouro','huginn'):
        depth=MODEL_PINS[model][2]
        cfg=json.loads((ROOT/'configs'/f'map_{model}_L512_decisive_start2.json').read_text())
        keep=[]
        for c in cfg['cohorts']:
            if c['regime'] not in ('P_pressure','verified_D_pressure') or c['cut'] not in (0,depth//2,depth):continue
            keep.append(c)
        cfg.update(name=f'map_{model}_L1536_subset_start0',repeat=0,seed=2026103481,
                   gpus=[0,1],cohorts=keep,dataset='prefix_1536.json',chunk=256,token_budget=256,
                   prompt_ids=[r['prompt_id'] for r in json.loads((ROOT/'data/prefix_1536.json').read_text())])
        dest=ROOT/'configs'/f'{cfg["name"]}.json'
        if dest.exists():raise FileExistsError(dest)
        write(dest,cfg);plans.append(dest.name)
    write(ROOT/'configs/long_subset_plan.json',dict(configs=plans,
          scope='Single-start endpoint/midpoint sensitivity at L1536, chunk256. Not a complete long-context cut optimum.',
          expansion='No online policy or nine-arm serving expansion without stage gates.'))
    print(' '.join(plans))


if __name__=='__main__':main()
