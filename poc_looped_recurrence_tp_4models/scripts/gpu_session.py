"""Reuse the audited GPU guard. Redirect only this process's artifact roots."""
from poc_looped_recurrence_tp_4models.contract import ROOT,PROJECT,ASSETS
from poc_looped_multigpu_kv_placement.scripts import gpu_session as shared
shared.ROOT=ROOT
shared.PROJECT=PROJECT
shared.ASSETS=ASSETS

if __name__=='__main__':shared.main()
