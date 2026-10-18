"""Reuse the publication check for only this task's candidate files."""
from poc_async_recurrent_pp.scripts import audit_publication as reused
from poc_decode_first_depth_aware_dp.contract import ROOT,PROJECT


if __name__=='__main__':
    reused.ROOT,reused.PROJECT=ROOT,PROJECT
    reused.main()
