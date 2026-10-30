"""Use the existing ownership checks in a separate resource ledger."""
from .common import ROOT,setup
from ..ouro_budget_joint_cg import resources as base
from ..adaptive_recurrence_cut.common import ROOT as PRIOR_ROOT

base.ROOT=ROOT
base.PRIOR=PRIOR_ROOT
base.APPROVED_LOGS.update({ROOT/'burner.log',PRIOR_ROOT/'burner.log',
    PRIOR_ROOT.parent/'rcc_pd/burner.log',PRIOR_ROOT.parent/'looped_pd_b_numerical_diagnosis/burner.log'})

def main():
    import argparse
    p=argparse.ArgumentParser();p.add_argument('action',choices=['status','stop','restore'])
    p.add_argument('--gpus',default='0,1,2,3,4,5,6,7');a=p.parse_args();setup()
    gpus=list(map(int,a.gpus.split(',')))
    if not set(gpus)<=set(range(8)):raise ValueError('Unauthorized device')
    if a.action=='status':print(dict(owned=base.burners(),active=base.processes()))
    else:getattr(base,a.action)(gpus)

if __name__=='__main__':main()
