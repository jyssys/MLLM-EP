"""Isolate Token fallback cost and Native wire progress with common credits."""
from .common import *
from .prepare import clone
from .matched_wire_cells import NATIVE_WIRE, fields
from .optimization_cells import FAST, paired


def main():
    initialize()
    jobs = []
    for arm in ('token_native_control', 'native_packed_burst'):
        source = 'secondary_huginn_token' if arm == 'token_native_control' else 'control_huginn_zero'
        options = fields('token') if arm == 'token_native_control' else dict(**FAST)
        options['native_transport_limits'] = dict(NATIVE_WIRE)
        if arm == 'token_native_control':
            options.update(minimum_P_pressure=10**9,
                qualification_allow_unpressured=False, source_monitor_boundaries=False,
                diagnostic_force_no_source_arming=True)
        else:
            options.update(wire_progress_burst=True, wire_progress_steps=64,
                wire_progress_budget_ms=5.)
            options['native_transport_limits'].update(transfer_chunk_bytes=512*1024**2,
                max_transfer_descriptors=4096)
        for count in (8, 32):
            name = f'matched_{arm}_state{count}'
            cfg = clone(f'{source}_state{count}', name,
                phase='MATCHED_NO_MIGRATION_CONTROL_EXACTNESS', **options)
            cfg['require_actual_offload_qualification'] = False
            if count == 32:
                cfg['required_qualification'] = f'matched_{arm}_state8'
            save(cfg)
            jobs.append(name)
        base = ('recovery_huginn_token_seed2026110931' if arm == 'token_native_control'
            else 'control_huginn_zero_seed2026110931')
        name = f'selection_matched_{arm}_seed2026110940'
        cfg = paired(clone(base, name, **options), name, f'matched_{arm}_state32')
        cfg['phase'] = 'MATCHED_CREDIT_CONTROL_SELECTION_NOT_OFFLOAD_SUCCESS'
        save(cfg)
        jobs.append(name)
    write(ROOT/'state/matched_additional_control_jobs.json', jobs)
    print(' '.join(jobs))


if __name__ == '__main__':
    main()
