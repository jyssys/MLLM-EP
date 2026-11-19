"""Compare clean/profile/clean standard-engine decode under fixed diagnostics."""
import json
import time
from .common import ROOT,save,write
from .run import execute


def main():
    declaration=ROOT/'configs/decode_profile_design_v1.json'
    if not declaration.exists():
        write(declaration,dict(time=time.time(),length=8192,output=16,concurrency=1,
            batch_invariant=[False,True],independent_starts_per_flag=3,
            order=['clean_before','profile','clean_after'],observer_gate_max_relative_error=.05,
            execution='Compile0, full decode CUDA graphs, synchronous scheduler. Same options in each clean/profile/clean triplet.',
            scope='Diagnostic control. Does not replace the stock continuous Main setting. No Layer/Token execution.'))
    for bi in (False,True):
        for label,profile in (('clean_before',False),('profile',True),('clean_after',False)):
            name=f'decode8192_BI{int(bi)}_{label}_C1_O16_v2'
            cfg=dict(name=name,method='Native',entry_module='poc_flashvep.dense_ar_native_reproduction.single_gpu',
                gpus=[1],batch_invariant=bi,lengths=[8192],async_scheduling=False,
                stock_execution_defaults=False,boundary_sync=True,CUDA_enclosure=True,
                kernel_activity=profile,diagnostic_only=True,wall_timeout=1200)
            assert execute(save(cfg)),name
    write(ROOT/'state/decode_profile_complete.json',dict(time=time.time(),complete=True))


if __name__=='__main__':main()
