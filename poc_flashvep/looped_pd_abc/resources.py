"""Use the existing ownership checks. Never signal unrelated GPU processes."""
import argparse
from pathlib import Path
from ..refinement_ep_allocation import resources as previous
from .common import ROOT,setup

base=previous.base
base.ROOT=ROOT
base.APPROVED_LOGS.update({ROOT/'burner.log',Path('/home/esjung/MLLM-EP-github/poc_flashvep/results/refinement_aware_ep_expert_allocation/burner.log')})
base.PRIOR=Path('/home/esjung/MLLM-EP-github/poc_flashvep/results/refinement_aware_ep_expert_allocation')

def main():
 setup();p=argparse.ArgumentParser();p.add_argument('action',choices=('status','stop','restore'))
 p.add_argument('--gpus',default='0,1,2,3,4,5,6,7');args=p.parse_args()
 gpus=list(map(int,args.gpus.split(',')));assert set(gpus)<=set(range(8))
 if args.action=='status':print(dict(owned=base.burners(),active=base.processes()))
 else:getattr(base,args.action)(gpus)

if __name__=='__main__':main()
