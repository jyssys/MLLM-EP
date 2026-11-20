"""Freeze the common corrected transport before new serving discovery."""
import time
from .common import ROOT,write
from .axis_qualification import main as qualify


def main():
    declaration=ROOT/'configs/frozen_shared_axis_runtime_v2.json'
    if not declaration.exists():
        write(declaration,dict(time=time.time(),B=8192,S=1024,KV_GiB=42,KV_block_tokens=128,
            restore_official_UCX_init=True,NIXL_threads=4,
            execution='Compile0, full decode graph, async=False',actual_batches=[1,1,1],source_credit=4,
            numerical_mode='BI0',natural_EOS=True,output_cap=2048,
            reason='KV128 passed full-state N8 and full200 task-quality checks. N32 descriptor/posting costs fell. '
            'Freeze before any new online axis-performance result. Default-KV16 and KV256 diagnostic evidence remain separate.',
            limits='Mechanism control. Not ordinary continuous-batch production scheduling. '
            'D quiescence is certified before suffix admission. No active-decode window insertion.'))
    qualify(tag='shared128_v2',cuts=(4,8,12,1,2,6),block=128,restore=True)


if __name__=='__main__':
    try:main()
    except BaseException as error:
        write(ROOT/'state/shared_axis_qualification_hold.json',dict(time=time.time(),error=repr(error)));raise
