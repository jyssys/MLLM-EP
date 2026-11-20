"""Separate GEMM efficiency from complete forward service and host launch."""
import time
from .common import ROOT,write,save
from .qualification import configuration
from .run import execute


def main():
    design=ROOT/'configs/frozen_BI0_GEMM_API_design_v1.json'
    if not design.exists():
        write(design,dict(time=time.time(),lengths=[4608,8064],cuts=[1,2,4,6,8,12],starts=3,
            primary_change='GEMM-only actual serving APIs, graph replay removes per-layer Python launch gaps.',
            hypothesis='The smallest Token M remains inefficient, while larger Token M approaches the large-M GEMM efficiency.',
            competing='Any complete-forward difference may instead be attention placement or non-GEMM/launch work.',
            scope='Primitive mechanism control. Not a serving speedup or a decode-safe fit result.'))
    for start in range(3):
        cfg=configuration(f'GEMM_API_BI0_Start{start}_v1','Native','longbench_shortanswer200_v2.json',count=1,gpus=[1],
            entry_module='poc_flashvep.dense_ar_bi0_shortanswer_pd.gemm_api_cost',
            state_capture=False,correctness_only=False,phase='ISOLATED_GEMM_API_CONTROL',wall_timeout=900)
        assert execute(save(cfg)),cfg['name']
    write(ROOT/'state/GEMM_API_complete.json',dict(time=time.time(),complete=True))


if __name__=='__main__':
    try:main()
    except BaseException as error:
        write(ROOT/'state/GEMM_API_hold.json',dict(time=time.time(),error=repr(error)));raise
