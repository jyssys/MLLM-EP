"""Use verified D pressure for independent repeated map starts."""
import json
from .common import ROOT,write


def main():
    plans=[]
    for model in ('ouro','huginn'):
        for repeat in (1,2):
            cfg=json.loads((ROOT/'configs'/f'map_{model}_L512_allcuts_start{repeat}_frozen.json').read_text())
            verified=json.loads((ROOT/'configs'/f'map_{model}_L512_verified_D_start{repeat}.json').read_text())
            depth=4 if model=='ouro' else 32
            cohorts=[c for c in cfg['cohorts'] if c['regime']!='D_pressure']+verified['cohorts']
            anchors=[c for c in cohorts if c['cut']==depth]
            candidates=[c for c in cohorts if c['cut']!=depth]
            cfg.update(name=f'map_{model}_L512_decisive_start{repeat}',cohorts=anchors+candidates)
            dest=ROOT/'configs'/f'{cfg["name"]}.json'
            if dest.exists():raise FileExistsError(dest)
            write(dest,cfg);plans.append(dest.name)
    write(ROOT/'configs/decisive_map_plan.json',dict(configs=plans,clean_parallel_workers=1,
          independent_starts=3,repeat0='Existing first map for P/balanced plus separate verified-D start0',
          reason='A fixed wall-time offset cannot prove that background Decode is active.'))
    print(' '.join(plans))


if __name__=='__main__':main()
