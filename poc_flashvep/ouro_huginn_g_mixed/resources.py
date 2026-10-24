"""Signal only verified burner sessions from approved experiment logs."""
from .common import ROOT, PRIOR
from ..ouro_budget_joint_cg import resources as base
base.APPROVED_LOGS.update({ROOT / 'burner.log', PRIOR / 'burner.log'})
base.ROOT, base.PRIOR = ROOT, PRIOR
processes, burners, stop, restore = base.processes, base.burners, base.stop, base.restore

if __name__ == '__main__':
    import sys
    devices = list(map(int, sys.argv[2].split(','))) if len(sys.argv) > 2 else list(range(8))
    if sys.argv[1] == 'status': print(dict(owned=burners(), active=processes()))
    else: {'stop': stop, 'restore': restore}[sys.argv[1]](devices)
