"""Independent 32-request qualification and explicit endpoint probes."""
import json
import random
from .common import *
from .prepare import base

def main():
    rows=json.loads((ROOT/'data/gate8.json').read_text());expanded=[]
    # Independent arrival order. Prompt content remains a finite eight-prompt fixture.
    order=list(range(8))*4;random.Random(2026103188).shuffle(order)
    for i,j in enumerate(order):
        row=dict(rows[j]);row.update(prompt_id=f'qualification_{i}_source_{j}',cap=4);expanded.append(row)
    write(ROOT/'data/qualification32.json',expanded)
    for model,gpus in [('ouro',[2,3]),('huginn',[6,7])]:
        for arm in ARMS:
            name=f'qualification32_{model}_{arm.lower()}_v1'
            cfg=base(model,arm,name,'qualification32.json',32,gpus)
            cfg.update(seed=2026103191,capacity=2,verify_transfer=True,state_probe=True,
                       cut=MODEL_PINS[model][2]//2 if arm.startswith('Loop') else MODEL_PINS[model][2],
                       chunk=256,token_budget=256,timeout=2400,arrival_offsets=[i*.013 for i in range(32)])
            if arm=='DP':cfg.update(chunk=4096,token_budget=4096,wavefront=False)
            write(ROOT/'configs'/f'{name}.json',cfg)
        for arm in ('Loop_Static','Token_Static'):
            for endpoint in ('zero','full'):
                name=f'endpoint8_{model}_{arm.lower()}_{endpoint}'
                cfg=base(model,arm,name,'gate8.json',8,gpus)
                cfg.update(verify_transfer=True,state_probe=True,seed=2026103101,
                    cut=0 if endpoint=='zero' else MODEL_PINS[model][2],
                    prefix_fraction=0. if endpoint=='zero' else 1.)
                write(ROOT/'configs'/f'{name}.json',cfg)

if __name__=='__main__':main()
