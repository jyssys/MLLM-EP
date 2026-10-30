"""Reuse audited burner ownership checks without changing the old ledger."""
import argparse
from .common import ROOT, PRIOR, setup
from ..ouro_budget_joint_cg import resources as base

base.ROOT = ROOT
base.PRIOR = PRIOR
base.APPROVED_LOGS.update({ROOT/'burner.log', PRIOR/'burner.log'})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=('stop', 'restore', 'status'))
    parser.add_argument('--gpus', default='0,1')
    args = parser.parse_args()
    setup()
    devices = list(map(int, args.gpus.split(',')))
    if not set(devices) <= set(range(8)):
        raise ValueError('Physical GPU outside authorized set')
    if args.action == 'status':
        print(dict(burners=base.burners(), active=base.processes()))
    else:
        getattr(base, args.action)(devices)


if __name__ == '__main__':
    main()
