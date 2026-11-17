"""Validate the source-macro8 regime suggested by the borrowed-cost screen."""
import fcntl
from .common import ROOT, QUIET_LOCK, resources, write
from .axis_configuration import base
from .discovery import job


def main():
    values = []
    try:
        for layers in (0, 2, 4, 8, 12):
            cfg = base(f'macro8_cal_Last{layers}_N64_O64_q4.8_v1',
                method='Native' if layers == 0 else 'Static', layers=layers,
                cap=64, matched=True, P_group_size=8, D_admission_batch=8,
                hidden_source_slots=32, hidden_destination_slots=32,
                D_combine_ready_macros=False, D_wait_for_source_macro=True)
            value = job(cfg)
            if value is not None:
                values.append(value)
                write(ROOT / 'analysis/macro8_calibration_serving.json', values)
    finally:
        with QUIET_LOCK.open('a+') as lease:
            fcntl.flock(lease, fcntl.LOCK_EX)
            if not resources().processes():
                resources().restore(list(range(8)))


if __name__ == '__main__':
    main()
