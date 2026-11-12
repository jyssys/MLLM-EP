"""Test the same explicit BF16 norm/position kernels on both query sizes."""
import fcntl,json
from .common import *
from .prepare import save
from .run import execute
from .analyze import compare


def main():
    jobs=[]
    for label,chunk in [('full',32000),('chunk256',256)]:
        c=json.loads((ROOT/'configs/chunk_numeric_full_N1_v1.json').read_text())
        c.update(name=f'chunk_explicit_norm_rope_{label}_N1_v1',token_budget=chunk,chunk=min(chunk,8000),
            ar_custom_ops=['none','+rms_norm','+rotary_embedding'],
            phase='SYMMETRIC_KERNEL_POLICY_DIAGNOSTIC')
        jobs.append(save(c))
    try:
        for n in jobs:
            if not execute(n,restore=False):return
        gate=compare(*jobs);write(ROOT/'analysis/chunk_explicit_kernel_output_gate.json',gate)
        print('EXPLICIT_KERNEL_GATE',gate['passed'],flush=True)
    finally:
        with (ROOT/'state/quiet_lane.lock').open('a+') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX)
            if not resources().processes():resources().restore(list(range(8)))


if __name__=='__main__':main()
