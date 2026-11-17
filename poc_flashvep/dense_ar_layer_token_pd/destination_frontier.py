"""Test whether layer results are limited by a small destination cohort."""
import fcntl
from .common import ROOT, QUIET_LOCK, resources, write
from .axis_configuration import base
from .discovery import job


def main():
    values = []
    try:
        for destination in (8, 16):
            for layers in (8, 12):
                cfg = base(f'fair_Layer_Last{layers}_P4_D{destination}_N128_q6.4_v1',
                    method='Static', count=128, rate=6.4, layers=layers,
                    D_admission_batch=destination, hidden_source_slots=32,
                    hidden_destination_slots=32, D_wait_for_source_macro=False,
                    D_combine_ready_macros=True)
                value = job(cfg)
                if value is not None:
                    values.append(value)
                    write(ROOT / 'analysis/layer_destination_frontier.json', values)
    finally:
        with QUIET_LOCK.open('a+') as lease:
            fcntl.flock(lease, fcntl.LOCK_EX)
            if not resources().processes():
                resources().restore(list(range(8)))


if __name__ == '__main__':
    main()
