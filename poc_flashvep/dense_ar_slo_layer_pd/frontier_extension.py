"""Test intermediate standard budgets, concurrency and routing fairly."""
import fcntl
from .common import *
from .configuration import base
from .run import execute
from .qualification import against_previous
from .analyze import metrics


def main():
    try:
        for budget,slots,placement in ((1024,2,'ROUND_ROBIN'),(2048,2,'ROUND_ROBIN'),
                (8192,2,'ROUND_ROBIN'),(8192,4,'ROUND_ROBIN'),
                (1024,2,'LEAST_OUTSTANDING'),(512,16,'LEAST_OUTSTANDING')):
            name=f'frontier_extend_C_B{budget}_S{slots}_{placement}_O16_q3.5_N128_v1'
            cfg=base(name,'Colocated',count=128,cap=16,rate=3.5,chunk=budget,
                token_budget=budget,D_slots=slots,placement=placement,
                verified_ByteLevel_priming=True,direct_phase_norm=True,ar_async_scheduling=True)
            assert execute(save(cfg),restore=False);assert against_previous(name)['passed']
            value=metrics(name);write(ROOT/'analysis'/(name+'_metrics.json'),value)
            print('FRONTIER_EXTENSION',name,value['TTFT_p95_ms'],value['ITL_p99_ms'],
                value['output_tokens_s'],flush=True)
        write(ROOT/'state/frontier_extension_complete.json',dict(time=time.time(),
            scope='Discovery configurations only. Tune each SLO objective independently before held-out confirmation.'))
    finally:
        with QUIET_LOCK.open('a+') as lease:
            fcntl.flock(lease,fcntl.LOCK_EX)
            if not resources().processes():resources().restore(list(range(8)))


if __name__=='__main__':main()
