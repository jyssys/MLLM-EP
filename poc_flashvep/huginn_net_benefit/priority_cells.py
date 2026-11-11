"""Qualify explicit Decode-first submission before claiming spatial headroom."""
from .common import *
from .prepare import clone
from .deferred_v2_candidates import fields
from .matched_wire_cells import NATIVE_WIRE
from .optimization_cells import paired


def main():
    initialize()
    jobs = []
    for sms in (None, 32):
        for arm in ('zero', 'loop31', 'token1984'):
            options = dict(fields('zero', bool(sms)), preserve_decode_cohort=True,
                native_transport_limits=NATIVE_WIRE, explicit_graph_teardown=True,
                child_exit_integrity_required=True, skip_dead_partial_coda_prepare=True,
                explicit_decode_priority=True, priority_decode_boundaries=True)
            if sms:
                options['prompt_green_sms'] = sms
            if arm == 'loop31':
                options.update(native_partial_cut=31, cut=31, method='loop', axis='loop',
                    atomic_prompt=False, forced_offload_diagnostic=True)
            if arm == 'token1984':
                options.update(native_frontier=[0, 1984], native_partial_cut=None,
                    cut=32, method='token', axis='token', D_prefill_loop_cap=8,
                    atomic_prompt=False, forced_offload_diagnostic=True)
            label = f'priority4_{"sm32" if sms else "normal"}_{arm}'
            for count in (8, 32):
                name = f'{label}_state{count}'
                cfg = clone(f'control_huginn_zero_state{count}', name,
                    phase='EXPLICIT_DECODE_FIRST_EXACTNESS_QUALIFICATION', **options)
                if count == 32:
                    cfg['required_qualification'] = f'{label}_state8'
                save(cfg)
                jobs.append(name)
            name = f'{label}_seed2026110940'
            cfg = paired(clone('control_huginn_zero_seed2026110931', name, **options),
                name, f'{label}_state32')
            cfg['phase'] = 'EXPLICIT_DECODE_FIRST_STATIC_SELECTION'
            save(cfg)
            jobs.append(name)
    write(ROOT/'state/explicit_decode_priority_jobs.json', jobs)
    print(' '.join(jobs))


if __name__ == '__main__':
    main()
