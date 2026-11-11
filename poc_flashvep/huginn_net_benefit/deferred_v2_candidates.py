"""Retain Native Decode cohort retirement and match the resolved wire policy."""
from .common import *
from .prepare import clone
from .optimization_cells import paired
from .deferred_candidates import fields
from .matched_wire_cells import NATIVE_WIRE


def main():
    initialize()
    jobs = []
    for green in (False, True):
        for arm in ('zero', 'loop31'):
            family = 'deferred2_green64' if green else 'deferred2'
            options = dict(fields(arm, green), preserve_decode_cohort=True,
                native_transport_limits=NATIVE_WIRE,
                skip_dead_partial_coda_prepare=True)
            for count in (8, 32):
                name = f'{family}_{arm}_state{count}'
                cfg = clone(f'control_huginn_zero_state{count}', name,
                    phase='DEFERRED_CODA_COHORT_RETIREMENT_MATCHED_WIRE_QUALIFICATION', **options)
                if count == 32:
                    cfg['required_qualification'] = f'{family}_{arm}_state8'
                save(cfg)
                jobs.append(name)
            name = f'{family}_{arm}_seed2026110940'
            cfg = paired(clone('control_huginn_zero_seed2026110931', name, **options),
                name, f'{family}_{arm}_state32')
            cfg['phase'] = 'DEFERRED_CODA_COHORT_RETIREMENT_SELECTION'
            save(cfg)
            jobs.append(name)
    write(ROOT/'state/deferred_coda_v2_jobs.json', jobs)
    print(' '.join(jobs))


if __name__ == '__main__':
    main()
