"""Freeze the initial constant-length, load-conditioned fixed-cut landscape."""
import json
import random
from .common import ROOT,CUTS,MODEL_PINS,write


def main():
    plans=[]
    for model in ('ouro','huginn'):
        depth=MODEL_PINS[model][2]
        source=json.loads((ROOT/'configs'/f'gate_{model}_L512_cut{depth}.json').read_text())
        for repeat in range(3):
            cohorts=[]
            cells=[(regime,cut) for regime in ('P_pressure','balanced','D_pressure') for cut in CUTS[model]]
            random.Random(20261030+repeat).shuffle(cells)
            for regime,cut in cells:
                if regime=='P_pressure':
                    caps=[16]*8;arrivals=[i*.001 for i in range(8)];cuts=[cut]*8;targets=[True]*8
                elif regime=='balanced':
                    caps=[64]*8;arrivals=[i*(.09 if model=='ouro' else .4) for i in range(8)];cuts=[cut]*8;targets=[True]*8
                else:
                    caps=[128]*2+[32]*6
                    delay=.15 if model=='ouro' else .6
                    arrivals=[0.,0.]+[delay+i*.002 for i in range(6)]
                    cuts=[depth]*2+[cut]*6;targets=[False]*2+[True]*6
                cohorts.append(dict(cohort_id=f'{regime}_cut{cut}',regime=regime,cut=cut,
                                    request_cuts=cuts,request_caps=caps,arrival_offsets=arrivals,target_mask=targets))
            name=f'map_{model}_L512_allcuts_start{repeat}'
            cfg=dict(source,name=name,cut=depth,cohorts=cohorts,
                     verify_transfer=False,state_probe=False,repeat=repeat,
                     gpus=[0,1] if repeat%2==0 else [4,5],
                     seed=2026103081+repeat*100,timeout=900,regime='fixed_map',warmup=False)
            dest=ROOT/'configs'/f'{name}.json'
            if dest.exists():raise FileExistsError(dest)
            write(dest,cfg);plans.append(dest.name)
    write(ROOT/'configs/static_map_allcuts_plan.json',dict(configs=plans,clean_parallel_workers=1,
          scope='8-request finite cohorts per cut and load regime, L512. Not sustainable capacity.',
          conditional_stages='No online or Token/Layer migration implementation before Stage-2 gate.'))
    print(' '.join(plans))


if __name__=='__main__':main()
