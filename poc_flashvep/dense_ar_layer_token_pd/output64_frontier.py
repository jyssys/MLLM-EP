"""Do not assume the output16 split remains optimal for output64."""
import fcntl
from .common import ROOT, QUIET_LOCK, resources, write
from .axis_configuration import base
from .discovery import job


def main():
    configs = []
    for suffix in (512,1024,2048,4096):
        configs.append(base(f'O64_Token_S{suffix}_P8_D8_N64_q4.8_v1',
            method='Token', count=64, cap=64, rate=4.8, suffix=suffix,
            matched=True, P_group_size=8, D_admission_batch=8))
    for destination in (8,16):
        for suffix in (1024,2048):
            configs.append(base(f'O64_Token_S{suffix}_P4_D{destination}_N64_q4.8_v1',
                method='Token', count=64, cap=64, rate=4.8, suffix=suffix,
                D_admission_batch=destination, D_wait_for_source_macro=False,
                D_combine_ready_macros=True))
    for layers in (2,4,8,12):
        configs.append(base(f'O64_Layer_Last{layers}_P4_D8_N64_q4.8_v1',
            method='Static', count=64, cap=64, rate=4.8, layers=layers,
            D_admission_batch=8, D_wait_for_source_macro=False,
            D_combine_ready_macros=True, hidden_source_slots=32,hidden_destination_slots=32))
    values = []
    try:
        for cfg in configs:
            value = job(cfg)
            if value is not None:
                values.append(value)
                write(ROOT / 'analysis/output64_axis_frontier.json',values)
    finally:
        with QUIET_LOCK.open('a+') as lease:
            fcntl.flock(lease, fcntl.LOCK_EX)
            if not resources().processes():
                resources().restore(list(range(8)))


if __name__ == '__main__':
    main()
