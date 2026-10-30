"""One additional independent start tests frozen choices, not another search."""
import json
from .common import ROOT,write


def main():
    frozen=json.loads((ROOT/'configs/frozen_regime_selections.json').read_text())['selections']
    plans=[]
    for model,selection in frozen.items():
        cfg=json.loads((ROOT/'configs'/f'map_{model}_L512_decisive_start2.json').read_text())
        depth=4 if model=='ouro' else 32
        keep=[]
        for c in cfg['cohorts']:
            if c['cut'] in (depth,selection['static_cut'],selection['regime_cuts'][c['regime']]):keep.append(c)
        cfg.update(name=f'map_{model}_L512_frozen_choices_start3',repeat=3,seed=2026103381,
                   gpus=[6,7],cohorts=keep)
        dest=ROOT/'configs'/f'{cfg["name"]}.json'
        if dest.exists():raise FileExistsError(dest)
        write(dest,cfg);plans.append(dest.name)
    write(ROOT/'configs/frozen_choice_holdout_plan.json',dict(configs=plans,
          rule='Only the start0-frozen static and regime-conditioned choices, plus full-PD anchors.',
          test_scope='Independent start/seed/GPU pair. The diagnostic prompt IDs remain the same. No new-domain generalization claim.'))
    print(' '.join(plans))


if __name__=='__main__':main()
