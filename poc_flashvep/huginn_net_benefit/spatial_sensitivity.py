"""Measure smaller prompt SM budgets under the same exact execution policy."""
from .common import *
from .prepare import clone
from .deferred_v2_candidates import fields
from .matched_wire_cells import NATIVE_WIRE
from .optimization_cells import paired


def main():
    initialize()
    jobs = []
    for sms in (16, 32):
        for arm in ('zero', 'loop31', 'token1984'):
            options = dict(fields('zero', True), prompt_green_sms=sms,
                preserve_decode_cohort=True, native_transport_limits=NATIVE_WIRE,
                skip_dead_partial_coda_prepare=True)
            if arm != 'zero':
                options.update(forced_offload_diagnostic=True, atomic_prompt=False)
            if arm == 'loop31':
                options.update(native_partial_cut=31, cut=31, method='loop', axis='loop')
            if arm == 'token1984':
                options.update(native_frontier=[0, 1984], native_partial_cut=None,
                    cut=32, method='token', axis='token', D_prefill_loop_cap=8)
            label = f'spatial_sm{sms}_{arm}'
            for count in (8, 32):
                name = f'{label}_state{count}'
                cfg = clone(f'control_huginn_zero_state{count}', name,
                    phase='SM_BUDGET_SENSITIVITY_EXACTNESS', **options)
                if count == 32:
                    cfg['required_qualification'] = f'{label}_state8'
                save(cfg)
                jobs.append(name)
            name = f'{label}_seed2026110940'
            cfg = paired(clone('control_huginn_zero_seed2026110931', name, **options),
                name, f'{label}_state32')
            cfg['phase'] = 'SM_BUDGET_SENSITIVITY_FORCED_STATIC_NOT_ADAPTIVE'
            save(cfg)
            jobs.append(name)
    write(ROOT/'state/spatial_sensitivity_jobs.json', jobs)
    print(' '.join(jobs))


if __name__ == '__main__':
    main()
