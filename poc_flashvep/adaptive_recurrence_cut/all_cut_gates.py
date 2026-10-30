"""A new independent start checks every cut before clean timing."""
import json
from .common import ROOT,CUTS,MODEL_PINS,write


def main():
    for model in ('ouro','huginn'):
        depth=MODEL_PINS[model][2]
        cfg=json.loads((ROOT/'configs'/f'gate_{model}_L128_cut{depth}_v3.json').read_text())
        name=f'gate_{model}_allcuts_L128'
        cohorts=[]
        for cut in reversed(CUTS[model]):
            cohorts.append(dict(cohort_id=f'cut{cut}',cut=cut,request_cuts=[cut]*8,
                                regime='all_cut_correctness',arrival_offsets=[i*.02 for i in range(8)]))
        cfg.update(name=name,cohorts=cohorts,gpus=[4,5] if model=='ouro' else [6,7],
                   seed=2026103051,warmup=False,verify_transfer=True,state_probe=True)
        dest=ROOT/'configs'/f'{name}.json'
        if dest.exists():raise FileExistsError(dest)
        write(dest,cfg)


if __name__=='__main__':main()
