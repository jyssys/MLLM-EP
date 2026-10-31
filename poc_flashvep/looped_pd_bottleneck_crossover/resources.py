"""Lease only verified owned burners. Never signal unrelated jobs."""
from .common import *
from ..ouro_budget_joint_cg import resources as base

PRIOR=REPO/'poc_flashvep/reports/adaptive_loop_token_pd_2gpu'
base.ROOT=ROOT;base.PRIOR=PRIOR
base.APPROVED_LOGS.update({ROOT/'burner.log',PRIOR/'burner.log',
    REPO/'poc_flashvep/reports/adaptive_recurrence_cut/burner.log'})

if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('action',choices=('status','stop','restore'));p.add_argument('--gpus',default='0,1,2,3,4,5,6,7');a=p.parse_args()
    gpus=list(map(int,a.gpus.split(',')));assert set(gpus)<=set(range(8));setup()
    if a.action=='status':print(dict(owned=base.burners(),active=base.processes()))
    else:getattr(base,a.action)(gpus)
