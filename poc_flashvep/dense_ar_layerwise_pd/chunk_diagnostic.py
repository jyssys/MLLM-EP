"""Localize a chunk output failure at the first shared query prefix."""
import fcntl,json
from .common import *
from .prepare import base,save
from .run import execute
from .analyze import compare


def main():
    jobs=[]
    for label,chunk,eager in [('full',32000,False),('chunk256',256,False),('chunk256_eager',256,True)]:
        c=base('chunk_numeric_'+label+'_N1_v1',count=1,cap=16)
        c.update(arm='DP2',atomic_prompt=False,chunk=min(chunk,8000),token_budget=chunk,
            layer_mode='STATE_REFERENCE',state_capture=True,state_capture_prefix_tokens=256,
            state_first_query_only=True,save_state_tensors=True,ar_enforce_eager=eager,
            fixed_KV_bytes=42*1024**3,calibration_serial=True,
            phase='CHUNK_NUMERIC_ROOT_CAUSE',clean_timing=False,timeout=240)
        jobs.append(save(c))
    results=[]
    try:
        for n in jobs:
            if not execute(n,restore=False):break
            gate=compare(jobs[0],n);results.append(gate)
            write(ROOT/'analysis/chunk_numeric_output_gates.json',dict(results=results))
            print('CHUNK_NUMERIC',n,gate['passed'],flush=True)
    finally:
        with (ROOT/'state/quiet_lane.lock').open('a+') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX)
            if not resources().processes():resources().restore(list(range(8)))


if __name__=='__main__':main()
