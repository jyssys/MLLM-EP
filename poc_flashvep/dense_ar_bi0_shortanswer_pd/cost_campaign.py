"""Qualify the isolated BI0 service harness before expanding its cost grid."""
import time
from .common import ROOT,write,save
from .qualification import configuration
from .run import execute


def job(name,lengths,cuts):
    cfg=configuration(name,'Native','longbench_shortanswer200_v2.json',count=1,
        gpus=[1],entry_module='poc_flashvep.dense_ar_bi0_shortanswer_pd.isolated_axis',
        state_capture=False,correctness_only=False,lengths=lengths,cuts=cuts,
        phase='ISOLATED_BI0_FORWARD_COST',cost_kernel_activity=True,repeats=12,
        wall_timeout=900)
    assert execute(save(cfg)),name


def main():
    design=ROOT/'configs/frozen_BI0_isolated_cost_design_v1.json'
    if not design.exists():
        write(design,dict(time=time.time(),diagnostic_lengths=[4608,8064],cuts=[1,2,4,6,8,12],
            actual_batch=1,logical_GEMM_rows='L*m == 36*s exactly',starts=3,
            execution='Stock BI0 decoder-layer APIs, eager, no CUDA graph. No transport or queue.',
            prefix_cache='Enabled only to create valid full-depth prefix KV for isolated Token continuation.',
            observer='Clean/profile/clean in the same process. Require <=5% change per cell before primary kernel breakdown.',
            caveat='These real-text prefixes are diagnostic inputs, not replacements for the frozen Natural200 benchmark.'))
    job('CostHarness_BI0_L4608_m1_start0_v1',[4608],[1])
    for start in range(3):
        job(f'CostGrid_BI0_L4608_8064_Start{start}_v1',[4608,8064],[1,2,4,6,8,12])
    write(ROOT/'state/isolated_cost_complete.json',dict(time=time.time(),complete=True))


if __name__=='__main__':
    try:main()
    except BaseException as error:
        write(ROOT/'state/isolated_cost_hold.json',dict(time=time.time(),error=repr(error)));raise
