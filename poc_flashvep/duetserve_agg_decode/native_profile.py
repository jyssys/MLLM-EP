"""Test a short P-only capture against isolated Native controls."""
import fcntl
import time

from .campaign import run_job
from .common import QUIET_LOCK, ROOT, resources, write
from .configuration import base


def main():
    try:
        for label in ('A1', 'HOST_EVENT', 'NSYS', 'A2'):
            extra = {}
            if label in ('HOST_EVENT', 'NSYS'):
                extra.update(iteration_trace=True, clean_timing=False)
            if label == 'NSYS':
                extra.update(nsys_bounded_capture=True, nsys_roles=['prefill'],
                             nsys_offset_s=10., nsys_seconds=2.)
            run_job(base(f'profile_Native_N128_O200_q6_s2026111616_{label}_v1',
                arm='PD_G1', count=128, rate=6., seed=2026111616, **extra))
        write(ROOT / 'state/native_profile_complete.json', dict(time=time.time(), status='COMPLETE'))
    finally:
        with QUIET_LOCK.open('a+') as lease:
            fcntl.flock(lease, fcntl.LOCK_EX)
            if not resources().processes():
                resources().restore(list(range(8)))


if __name__ == '__main__':
    main()
