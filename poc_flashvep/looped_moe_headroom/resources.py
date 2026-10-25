"""Lease only the verified burner processes from approved experiment logs."""
from .common import ROOT, PRIOR
from ..ouro_budget_joint_cg import resources as base

base.ROOT, base.PRIOR = ROOT, PRIOR
base.APPROVED_LOGS.update({ROOT / 'burner.log', PRIOR / 'burner.log'})
processes, burners, stop, restore = base.processes, base.burners, base.stop, base.restore

if __name__ == '__main__':
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('action', choices=['status', 'stop', 'restore'])
    p.add_argument('--gpus', default='0,1,2,3,4,5,6,7')
    a = p.parse_args()
    devices = list(map(int, a.gpus.split(',')))
    assert set(devices) <= set(range(8))
    if a.action == 'status':
        print(dict(owned=burners(), active=processes()))
    else:
        {'stop': stop, 'restore': restore}[a.action](devices)
