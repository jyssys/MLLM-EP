"""Reuse the validated exact-mask burner guard without model changes."""
from poc_async_recurrent_pp.scripts import gpu_session as reused
from poc_recurrent_asymmetry.contract import ROOT,PROJECT,ASSETS


if __name__=='__main__':
    reused.ROOT,reused.PROJECT,reused.ASSETS=ROOT,PROJECT,ASSETS
    reused.main()
