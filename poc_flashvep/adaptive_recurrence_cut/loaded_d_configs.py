"""Freeze probes that require live background Decode before target arrival."""
import json
from .common import ROOT,write


def main():
    plans=[]
    for model in ('ouro','huginn'):
        for repeat in range(3):
            path=ROOT/'configs'/f'map_{model}_L512_allcuts_start{repeat}_frozen.json'
            cfg=json.loads(path.read_text())
            cfg['name']=f'map_{model}_L512_verified_D_start{repeat}'
            cfg['cohorts']=[dict(c,regime='verified_D_pressure',cohort_id=c['cohort_id'].replace('D_pressure','verified_D_pressure'),
                                 ready_background_requests=2,background_ready_tokens=4)
                            for c in cfg['cohorts'] if c['regime']=='D_pressure']
            dest=ROOT/'configs'/f'{cfg["name"]}.json'
            if dest.exists():raise FileExistsError(dest)
            write(dest,cfg);plans.append(dest.name)
    write(ROOT/'configs/verified_D_pressure_plan.json',dict(configs=plans,
          condition='Both background requests must deliver at least four output tokens and remain active before any target arrives.',
          limitation='State-triggered mechanism cohorts. Not an open-loop capacity result.',clean_parallel_workers=1))
    print(' '.join(plans))


if __name__=='__main__':main()
