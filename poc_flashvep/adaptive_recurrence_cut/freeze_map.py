"""Place parity anchors first. Freeze the measurement order before execution."""
import json
from .common import ROOT,write


def main():
    plan=json.loads((ROOT/'configs/static_map_allcuts_plan.json').read_text())
    paths=[]
    for source in plan['configs']:
        cfg=json.loads((ROOT/'configs'/source).read_text());depth=4 if cfg['model']=='ouro' else 32
        anchors=[c for c in cfg['cohorts'] if c['cut']==depth]
        candidates=[c for c in cfg['cohorts'] if c['cut']!=depth]
        cfg['cohorts']=anchors+candidates;cfg['enforce_output_parity']=True
        cfg['name']+='_frozen'
        dest=ROOT/'configs'/f'{cfg["name"]}.json'
        if dest.exists():raise FileExistsError(dest)
        write(dest,cfg);paths.append(dest.name)
    write(ROOT/'configs/static_map_frozen_plan.json',dict(configs=paths,clean_parallel_workers=1))
    print(' '.join(paths))


if __name__=='__main__':main()
