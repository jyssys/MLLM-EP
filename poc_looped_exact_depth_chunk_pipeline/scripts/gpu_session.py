"""Reuse scoped stop/run/restore handling. Do not affect other GPUs."""
from poc_looped_exact_depth_chunk_pipeline.contract import ROOT,PROJECT,ASSETS
from poc_looped_multigpu_kv_placement.scripts import gpu_session as shared
shared.ROOT=ROOT
shared.PROJECT=PROJECT
shared.ASSETS=ASSETS
if __name__=='__main__':shared.main()
