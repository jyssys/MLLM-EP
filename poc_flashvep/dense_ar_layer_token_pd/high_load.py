"""Use a genuinely backlogged source before interpreting capacity differences."""
import fcntl
from .common import QUIET_LOCK, ROOT, resources, write
from .axis_configuration import base
from .discovery import job


def main():
    jobs = []
    for group in (4, 8):
        for layers in (8, 12):
            jobs.append(base(f'high_Static_Last{layers}_P{group}_D4_N128_O16_q6.4_v1',
                method='Static', count=128, rate=6.4, layers=layers,
                P_group_size=group, hidden_source_slots=32, hidden_destination_slots=32,
                D_admission_batch=4, D_wait_for_source_macro=False, D_combine_ready_macros=True))
    for suffix in (1024, 2048, 4096):
        jobs.append(base(f'high_Token_S{suffix}_P4_D16_N128_O16_q6.4_v1',
            method='Token', count=128, rate=6.4, suffix=suffix,
            D_admission_batch=16, D_wait_for_source_macro=False, D_combine_ready_macros=True))
    try:
        values = []
        for cfg in jobs:
            value = job(cfg)
            if value is not None:
                values.append(value)
                write(ROOT / 'analysis/high_load_O16_axis_screen.json', values)
    finally:
        with QUIET_LOCK.open('a+') as lease:
            fcntl.flock(lease, fcntl.LOCK_EX)
            if not resources().processes():
                resources().restore(list(range(8)))


if __name__ == '__main__':
    main()
