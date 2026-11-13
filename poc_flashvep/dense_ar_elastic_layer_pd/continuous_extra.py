"""Check intermediate chunks and routing before freezing a strong baseline."""
import fcntl
from .common import *
from .configuration import base
from .run import execute
from .analyze import compare,metrics


def main():
    rows=[]
    try:
        for cap in (16,64,200):
            for budget,placement in ((384,'ROUND_ROBIN'),(768,'ROUND_ROBIN'),(512,'LEAST_OUTSTANDING')):
                name=f'continuous_extra_B{budget}_{placement}_O{cap}_N64_v1'
                cfg=base(name,arm='Colocated',cap=cap,token_budget=budget,chunk=budget,
                    placement=placement,phase='STRONG_CONTINUOUS_INTERMEDIATE_CHUNK_AND_ROUTING_DISCOVERY')
                if not execute(save(cfg),restore=False):return
                output=compare('baseline_Native_O200_N64_q4p8_s21_v1',name,True)
                row=dict(name=name,output=output,metrics=metrics(name));rows.append(row)
                write(ROOT/'state/continuous_extra_progress.json',dict(results=rows))
                print('CONTINUOUS_EXTRA',name,output['passed'],*[round(row['metrics'][k],2) for k in ('TTFT_p95_ms','ITL_p99_ms','TPOT_p95_ms')],flush=True)
                if not output['passed']:return
        write(ROOT/'analysis/continuous_extra_results.json',dict(results=rows,
            scope='Viewed selection prompts. Different chunk sizes are not independent starts.'))
    finally:
        with QUIET_LOCK.open('a+') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX)
            if not resources().processes():resources().restore(list(range(8)))


if __name__=='__main__':main()
