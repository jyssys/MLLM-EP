"""Test a generic deferred-log correction without moving delivery timestamps."""
import fcntl
from .common import *
from .configuration import base
from .run import execute
from .analyze import compare,metrics


def main():
    rows=[]
    try:
        for arm in ('Native','Revised','Colocated'):
            for deferred in ((True,False) if arm!='Colocated' else (True,)):
                cfg=base(f'completion_log_{arm}_defer{int(deferred)}_O16_N64_v1',arm=arm,cap=16,
                    defer_completion_log=deferred,completion_handler_timing=True,
                    phase='GENERIC_COMPLETION_RECORD_IO_CONTROL')
                if arm=='Colocated':cfg.update(token_budget=512,chunk=512)
                if not execute(save(cfg),restore=False):return
                output=compare('baseline_Native_O200_N64_q4p8_s21_v1',cfg['name'],True)
                row=dict(name=cfg['name'],output=output,metrics=metrics(cfg['name']));rows.append(row)
                write(ROOT/'state/completion_log_progress.json',dict(results=rows))
                print('COMPLETION_LOG',cfg['name'],output['passed'],
                    *[round(row['metrics'][k],2) for k in ('TTFT_p95_ms','ITL_p99_ms','TPOT_p95_ms','worker_emission_ITL_p99_ms')],flush=True)
                if not output['passed']:return
        write(ROOT/'analysis/completion_log_control.json',dict(results=rows,
            scope='Generic output-delivery optimization. All primary methods must share the adopted setting. Detailed records retain their original delivery timestamps and flush after worker shutdown.'))
    finally:
        with QUIET_LOCK.open('a+') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX)
            if not resources().processes():resources().restore(list(range(8)))


if __name__=='__main__':main()
