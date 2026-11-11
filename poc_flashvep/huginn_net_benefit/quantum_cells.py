"""Allow both methods to tune execution quantum independently of the cut."""
from .common import *
from .prepare import clone
from .deferred_v2_candidates import fields
from .matched_wire_cells import NATIVE_WIRE
from .optimization_cells import paired


def main():
    initialize()
    jobs = []
    variants = [('token1984', quantum) for quantum in (1, 2, 4, 16, 32)]
    variants += [('loop31', tile) for tile in (64, 128, 256, 512, 1024)]
    for arm, size in variants:
        options = dict(fields('zero', False), preserve_decode_cohort=True,
            native_transport_limits=NATIVE_WIRE, explicit_graph_teardown=True,
            child_exit_integrity_required=True, skip_dead_partial_coda_prepare=True,
            explicit_decode_priority=True, priority_decode_boundaries=True,
            atomic_prompt=False, forced_offload_diagnostic=True)
        if arm == 'token1984':
            options.update(native_frontier=[0, 1984], native_partial_cut=None,
                cut=32, method='token', axis='token', D_prefill_loop_cap=size)
        else:
            options.update(native_partial_cut=31, cut=31, method='loop', axis='loop',
                D_prefill_loop_cap=1, D_prefill_token_cap=size)
        label = f'quantum6_{arm}_{size}'
        for count in (8, 32):
            name = f'{label}_state{count}'
            cfg = clone(f'control_huginn_zero_state{count}', name,
                phase='STATIC_QUANTUM_AND_DEPENDENCY_QUALIFICATION', **options)
            if count == 32:
                cfg['required_qualification'] = f'{label}_state8'
            save(cfg)
            jobs.append(name)
        name = f'{label}_seed2026110940'
        cfg = paired(clone('control_huginn_zero_seed2026110931', name, **options),
            name, f'{label}_state32')
        cfg['phase'] = 'STATIC_QUANTUM_SELECTION_NOT_ADAPTIVE_CUT_PROOF'
        save(cfg)
        jobs.append(name)
    write(ROOT/'state/static_quantum_sensitivity_jobs.json', jobs)
    print(' '.join(jobs))


if __name__ == '__main__':
    main()
