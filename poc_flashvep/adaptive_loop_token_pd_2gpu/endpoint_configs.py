"""Endpoint checks use pairs outside the expanded qualification jobs."""
import json
from .common import ROOT,write

def main():
    for model,gpus in [('ouro',[0,1]),('huginn',[4,5])]:
        for arm in ('Loop_Static','Token_Static'):
            for endpoint in ('zero','full'):
                name=f'endpoint8_{model}_{arm.lower()}_{endpoint}'
                cfg=json.loads((ROOT/'configs'/f'{name}.json').read_text())
                cfg.update(name=name+'_pairs_v2',gpus=gpus)
                dest=ROOT/'configs'/f'{cfg["name"]}.json'
                if dest.exists():raise FileExistsError(dest)
                write(dest,cfg)

if __name__=='__main__':main()
