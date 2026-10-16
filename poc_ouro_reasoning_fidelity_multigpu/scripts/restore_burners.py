"""Run the scoped launcher after all experiment jobs finish."""
from poc_ouro_reasoning_fidelity_multigpu.contract import ROOT,PROJECT,ASSETS,GPUS
from poc_looped_multigpu_kv_placement.scripts import restore_burners as shared
shared.ROOT,shared.PROJECT,shared.ASSETS=ROOT,PROJECT,ASSETS
shared.GPUS=GPUS
shared.MASK=','.join(map(str,GPUS))
if __name__=='__main__':
    shared.main()
