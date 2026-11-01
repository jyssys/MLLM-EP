"""Reuse ownership-checked GPU leases with a new ledger and explicit restoration set."""
from .common import *
from ..ouro_budget_joint_cg import resources as inherited

inherited.ROOT=ROOT;inherited.PRIOR=PRIOR
inherited.APPROVED_LOGS.update({ROOT/'burner.log',PRIOR/'burner.log'})


if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('action',choices=('status','stop','restore'))
    p.add_argument('--gpus',default='0,1,2,3,4,5,6,7');a=p.parse_args();setup()
    gpus=list(map(int,a.gpus.split(',')));assert set(gpus)<=set(range(8))
    if a.action=='status':print(dict(owned=inherited.burners(),active=inherited.processes()))
    else:getattr(inherited,a.action)(gpus)
