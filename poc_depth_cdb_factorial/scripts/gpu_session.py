"""Reuse the ownership guard and restore only the requested burners."""
from poc_async_recurrent_pp.scripts import gpu_session as reused
from poc_depth_cdb_factorial.contract import ROOT, PROJECT, ASSETS

if __name__ == '__main__':
    reused.ROOT, reused.PROJECT, reused.ASSETS = ROOT, PROJECT, ASSETS
    reused.main()
