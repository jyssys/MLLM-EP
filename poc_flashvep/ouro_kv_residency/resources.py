"""Reuse ownership checks without changes to previous experiment files."""
from .common import ROOT,PRIOR
from ..ouro_budget_joint_cg import resources as inherited
inherited.APPROVED_LOGS.update({ROOT/'burner.log',PRIOR/'burner.log'})
inherited.ROOT=ROOT
inherited.PRIOR=PRIOR
processes=inherited.processes
burners=inherited.burners
stop=inherited.stop
restore=inherited.restore

if __name__=='__main__':
    import sys
    devices=list(map(int,sys.argv[2].split(','))) if len(sys.argv)>2 else list(range(8))
    assert set(devices)<=set(range(8))
    if sys.argv[1]=='status':print(dict(owned=burners(),active=processes()))
    else:{'stop':stop,'restore':restore}[sys.argv[1]](devices)
