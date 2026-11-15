"""Run bounded Colocated discovery with exact output gates."""
import argparse,fcntl
from .common import *
from .configuration import base
from .run import execute
from .qualification import against_previous
from .analyze import metrics


def initial():
    jobs=[base('qual_Native_N8_O16_v1',count=8),
        base('qual_Colocated_N32_B128_O16_v1','Colocated',count=32,chunk=128,token_budget=128,D_slots=16)]
    for budget in (64,128,256,512):
        jobs.append(base(f'frontier_Colocated_N64_B{budget}_S16_O16_q3_v1','Colocated',
            chunk=budget,token_budget=budget,D_slots=16))
    jobs.append(base('frontier_Colocated_N64_B8192_S32_O16_q4p8_v1','Colocated',
        rate=4.8,chunk=8192,token_budget=8192,D_slots=32))
    return jobs


def run(jobs):
    initialize()
    try:
        for cfg in jobs:
            name=save(cfg)
            if not execute(name,restore=False):raise RuntimeError('GPU run failed: '+name)
            gate=against_previous(name)
            write(ROOT/'analysis'/(name+'_metrics.json'),metrics(name))
            if not gate['passed']:raise RuntimeError('Output gate failed: '+name)
            append(ROOT/'state/discovery_completed.jsonl',dict(time=time.time(),name=name,status='OUTPUT_EXACT'))
    finally:
        with QUIET_LOCK.open('a+') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX)
            if not resources().processes():resources().restore(list(range(8)))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--initial',action='store_true');args=p.parse_args()
    if args.initial:run(initial())
