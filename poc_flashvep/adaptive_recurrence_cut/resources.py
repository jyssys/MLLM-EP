"""Reuse the owned-burner lease checks without changing prior ledgers."""
import argparse
from .common import ROOT,PRIOR,NUMERICAL,setup
from ..ouro_budget_joint_cg import resources as base

base.ROOT=ROOT
base.PRIOR=PRIOR
base.APPROVED_LOGS.update({ROOT/'burner.log',PRIOR/'burner.log',NUMERICAL/'burner.log'})


def main():
    p=argparse.ArgumentParser();p.add_argument('action',choices=('status','stop','restore'))
    p.add_argument('--gpus',default='0,1,2,3,4,5,6,7');a=p.parse_args();setup()
    devices=list(map(int,a.gpus.split(',')))
    if not set(devices)<=set(range(8)):raise ValueError('Unauthorized physical GPU')
    if a.action=='status':print(dict(owned_burners=base.burners(),active=base.processes()))
    else:getattr(base,a.action)(devices)


if __name__=='__main__':main()
