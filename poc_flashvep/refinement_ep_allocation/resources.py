"""Reuse ownership checks for this campaign's burner leases."""
import argparse
import time
from pathlib import Path
from .common import ROOT, setup
from ..ouro_budget_joint_cg import resources as base

base.ROOT = ROOT
base.PRIOR = Path('/home/esjung/MLLM-EP-github/poc_flashvep/reports/looped_moe_ep_affinity_placement')
base.APPROVED_LOGS.update({ROOT/'burner.log', base.PRIOR/'burner.log'})
_audited_stop=base.stop

def retry_owned_stop(gpus):
    """Retry only a vanished process snapshot. Keep every ownership assertion."""
    for attempt in range(10):
        try:
            return _audited_stop(gpus)
        except (FileNotFoundError,ProcessLookupError):
            if attempt==9:
                raise
            time.sleep(1)

base.stop=retry_owned_stop

def main():
    setup()
    p = argparse.ArgumentParser()
    p.add_argument('action', choices=('status', 'stop', 'restore'))
    p.add_argument('--gpus', default='0,1,2,3,4,5,6,7')
    a = p.parse_args()
    gpus = list(map(int, a.gpus.split(',')))
    assert set(gpus) <= set(range(8))
    if a.action == 'status':
        print(dict(owned_burners=base.burners(), active=base.processes()))
    else:
        getattr(base, a.action)(gpus)

if __name__ == '__main__':
    main()
