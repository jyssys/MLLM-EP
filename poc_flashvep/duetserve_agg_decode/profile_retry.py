"""Repeat only captures rejected by a documented profiler-session guard mismatch."""
import fcntl
import json
import time

from .campaign import run_job
from .common import QUIET_LOCK, ROOT, initialize, resources, write
from .configuration import base, standard_capture_sizes


def main():
    initialize()
    until = time.monotonic() + 3600
    while not (ROOT / 'state/profile_phase_complete.json').exists():
        assert time.monotonic() < until
        time.sleep(5)
    try:
        for budget, slots in ((8192, 1024), (512, 16)):
            previous = f'profile_DP2_B{budget}_S{slots}_FULL_N128_O200_q6_s2026111616_NSYS_v1'
            receipt = json.loads((ROOT / 'raw' / previous / 'dispatcher_result.json').read_text())
            if receipt['complete']:
                continue
            assert receipt['returncode'] == 0 and receipt['normal_worker_exit']
            observations = json.loads(next((ROOT / 'raw' / previous).glob('NVML-summary-*.json')).read_text())
            ready = json.loads((ROOT / 'raw' / previous / 'replica0_ready.json').read_text())
            # Preserve the failed original receipt. Only the known Nsight
            # worker session mismatch permits this narrowly scoped retry.
            assert {r['pid'] for r in observations['foreign_process_observations']} == {ready['pid']}
            name = previous.removesuffix('_v1') + '_v2'
            extra = dict(iteration_trace=True, clean_timing=False,
                nsys_bounded_capture=True, nsys_roles=['replica0'], nsys_offset_s=10., nsys_seconds=2.)
            if slots == 1024:
                extra['explicit_cudagraph_capture_sizes'] = standard_capture_sizes(1024)
            run_job(base(name, budget=budget, slots=slots, count=128, rate=6., seed=2026111616, **extra))
        write(ROOT / 'state/nsys_retry_complete.json', dict(time=time.time(), status='COMPLETE'))
    finally:
        with QUIET_LOCK.open('a+') as lease:
            fcntl.flock(lease, fcntl.LOCK_EX)
            if not resources().processes():
                resources().restore(list(range(8)))


if __name__ == '__main__':
    main()
