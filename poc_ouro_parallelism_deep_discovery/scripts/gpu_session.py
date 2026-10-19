"""Use the existing tested ownership guard in a separate campaign namespace."""
from poc_async_recurrent_pp.scripts import gpu_session as reused
from poc_ouro_parallelism_deep_discovery.contract import ROOT, PROJECT, ASSETS

if __name__=='__main__':
    reused.ROOT,reused.PROJECT,reused.ASSETS=ROOT,PROJECT,ASSETS
    reused.main()
