"""Reuse the scoped GPU guard and unconditional burner restoration."""
from poc_async_recurrent_pp.scripts import gpu_session as reused
from poc_decode_first_depth_aware_dp.contract import ROOT,PROJECT,ASSETS


if __name__=='__main__':
    reused.ROOT,reused.PROJECT,reused.ASSETS=ROOT,PROJECT,ASSETS
    reused.main()
