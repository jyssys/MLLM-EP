"""Reuse verified burner ownership checks, restricted to physical GPUs 4–7."""
from .common import ROOT, PRIOR, GPUS
from ..ouro_budget_joint_cg import resources as inherited

inherited.APPROVED_LOGS = inherited.APPROVED_LOGS | {ROOT/'burner.log', PRIOR/'burner.log'}
inherited.ROOT = ROOT
inherited.PRIOR = PRIOR
processes = inherited.processes
burners = inherited.burners

def stop(gpus):
    assert set(gpus) <= set(GPUS), ('Forbidden GPU', gpus)
    inherited.stop(gpus)

def restore(gpus):
    assert set(gpus) <= set(GPUS), ('Forbidden GPU', gpus)
    inherited.restore(gpus)

if __name__ == '__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('action',choices=['status','stop','restore'])
    p.add_argument('--gpus',default='4,5,6,7');a=p.parse_args()
    if a.action=='status':
        print(dict(owned_burners={g:p for g,p in burners().items() if g in GPUS},
            active=[r for r in processes() if r['gpu'] in GPUS]))
    else:
        globals()[a.action](list(map(int,a.gpus.split(','))))
