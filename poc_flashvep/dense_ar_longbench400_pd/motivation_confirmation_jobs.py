"""Repeat the three QPS points that were declared before the main sweep."""
import copy
import json
import numpy as np
from .common import ROOT,write
from .run import save
from .next_jobs import plan


def main():
    jobs=[];rows=[]
    for qps in (1.8,2.8,4.5):
        for seed in (2026112162,2026112163):
            for method in (('Agg','Native') if seed%2 else ('Native','Agg')):
                old=f'Motive400_{method}_q{qps}_seed2026112161_v1'
                cfg=copy.deepcopy(json.loads((ROOT/'configs'/(old+'.json')).read_text()))
                cfg.update(name=f'Motive400_{method}_q{qps}_seed{seed}_confirm_v1',seed=seed,
                    arrival_offsets=np.r_[0.,np.cumsum(np.random.default_rng(seed).exponential(1/qps,cfg['count']-1))].tolist(),
                    confirmation=True,discovery_only=False,
                    repetition_scope='The QPS points1.8,2.8,4.5 were predeclared before this confirmation. '
                        'The discovery start and two additional independent server-start/arrival-seed blocks form n3. '
                        'Same prompts and trace are paired across Agg/Native within each seed. '
                        'Use the original immutable default-compile baseline source capsule.')
                jobs.append(save(cfg));rows.append(dict(name=cfg['name'],method=method,QPS=qps,seed=seed))
    write(ROOT/'configs/motivation_independent_confirmation_v1.json',dict(rows=rows,
        discovery_seed=2026112161,additional_seeds=[2026112162,2026112163],
        source='Existing motivation_baseline_v1 capsule. No new adapter or observer changes.',
        scope='Finite400 cohort with drain. Not a sustainable-capacity measurement.'))
    plan('motivation400_confirmation_v1',jobs)
    print('Frozen12 additional baseline starts at three predeclared QPS points.',flush=True)


if __name__=='__main__':main()
