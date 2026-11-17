"""Give both axes the same legal source-macro8 alternative."""
import fcntl
from .common import ROOT, QUIET_LOCK, resources, write
from .axis_configuration import base
from .discovery import job


def main():
    configs = [base('macro8_Last12_N128_O16_q6.4_v1', method='Static', count=128,
        layers=12, rate=6.4, matched=True, P_group_size=8, D_admission_batch=8,
        hidden_source_slots=32, hidden_destination_slots=32)]
    configs.extend(base(f'macro8_Token{suffix}_N128_O16_q6.4_v1', method='Token',
        count=128, rate=6.4, suffix=suffix, matched=True, P_group_size=8,
        D_admission_batch=8) for suffix in (2048,2672,4096))
    values = []
    try:
        for cfg in configs:
            value = job(cfg)
            if value is not None:
                values.append(value)
                write(ROOT / 'analysis/macro8_axis_frontier.json', values)
    finally:
        with QUIET_LOCK.open('a+') as lease:
            fcntl.flock(lease, fcntl.LOCK_EX)
            if not resources().processes():
                resources().restore(list(range(8)))


if __name__ == '__main__':
    main()
