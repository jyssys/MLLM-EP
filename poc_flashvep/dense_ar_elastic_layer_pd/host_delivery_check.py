"""Test whether coordinator thread scheduling explains delivery tails."""
import fcntl
from .common import *
from .configuration import base
from .run import execute
from .analyze import compare,metrics


def main():
    rows=[]
    try:
        for arm in ('Native','Revised'):
            for variant,interval in (('A',.005),('B',.001),('C',.005)):
                cfg=base(f'host_switch_{arm}_O16_N64_{variant}_v1',arm=arm,cap=16,
                    coordinator_GIL_switch_interval_s=interval,
                    phase='ONE_FACTOR_HOST_DELIVERY_ABA_DIAGNOSIS')
                if not execute(save(cfg),restore=False):break
                output=compare('baseline_Native_O200_N64_q4p8_s21_v1',cfg['name'],True)
                row=dict(name=cfg['name'],output=output,metrics=metrics(cfg['name']))
                rows.append(row);write(ROOT/'state/host_delivery_progress.json',dict(results=rows))
                print('HOST_SWITCH',cfg['name'],output['passed'],row['metrics']['ITL_p99_ms'],
                    row['metrics']['worker_emission_ITL_p99_ms'],row['metrics']['coordinator_delivery_lag_p95_ms'],flush=True)
                if not output['passed']:return
        write(ROOT/'analysis/host_delivery_thread_control.json',dict(results=rows,
            scope='One-factor A/B/A diagnosis. Worker intervals include GPU and Host work. This does not measure exclusive kernel time.'))
    finally:
        with QUIET_LOCK.open('a+') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX)
            if not resources().processes():resources().restore(list(range(8)))


if __name__=='__main__':main()
