from .common import ROOT,PRIOR
from ..ouro_budget_joint_cg import resources as base
base.ROOT,base.PRIOR=ROOT,PRIOR
base.APPROVED_LOGS.update({ROOT/'burner.log',PRIOR/'burner.log'})
processes,burners,stop,restore=base.processes,base.burners,base.stop,base.restore
