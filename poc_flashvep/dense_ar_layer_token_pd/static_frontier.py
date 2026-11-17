"""Tune source macro geometry as well as the layer count."""
import fcntl
import json
from .common import QUIET_LOCK, ROOT, resources, write
from .axis_configuration import base
from .discovery import job


def main():
    jobs = []
    for group in (4, 8, 16):
        for layers in (8, 12):
            name = f'frontier_Static_Last{layers}_Pmacro{group}_D4_N64_O16_q4.8_v2'
            jobs.append(base(name, method='Static', layers=layers,
                P_group_size=group, hidden_source_slots=32, hidden_destination_slots=32,
                D_admission_batch=4, D_wait_for_source_macro=False, D_combine_ready_macros=True,
                P_collect_ms=2))
    try:
        values = []
        for cfg in jobs:
            value = job(cfg)
            if value is not None:
                values.append(value)
                write(ROOT / 'analysis/static_macro_frontier_discovery.json', values)
    finally:
        with QUIET_LOCK.open('a+') as lease:
            fcntl.flock(lease, fcntl.LOCK_EX)
            if not resources().processes():
                resources().restore(list(range(8)))


if __name__ == '__main__':
    main()
