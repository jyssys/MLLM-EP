"""Strong large-suffix Token controls and deeper Loop cuts under one Runtime."""
from .common import *
from .prepare import clone
from .optimization_cells import paired
from .deferred_v2_candidates import fields
from .matched_wire_cells import NATIVE_WIRE


def main():
    initialize()
    names = []
    # A cut can be screened out after a harmful serving result. Preparation
    # does not authorize treating an unexecuted arm as a zero or a failure.
    for kind, point in (('token', 1984), ('token', 1792), ('token', 1536),
            ('token', 1024), ('loop', 30), ('loop', 28), ('loop', 24)):
        for green in (False, True):
            label = f'static2_{"green64_" if green else ""}{kind}{point}'
            options = dict(fields('zero', green), native_transport_limits=NATIVE_WIRE,
                preserve_decode_cohort=True, skip_dead_partial_coda_prepare=True,
                D_prefill_loop_cap=8, forced_offload_diagnostic=True,
                atomic_prompt=False, method=kind, axis=kind)
            if kind == 'token':
                options.update(native_frontier=[0, point], native_partial_cut=None,
                    cut=32, D_token_budget=2048)
            else:
                options.update(native_partial_cut=point, cut=point, native_frontier=None)
            for count in (8, 32):
                name = f'{label}_state{count}'
                cfg = clone(f'control_huginn_zero_state{count}', name,
                    phase='MATCHED_NATIVE_STATIC_GRANULARITY_QUALIFICATION', **options)
                if count == 32:
                    cfg['required_qualification'] = f'{label}_state8'
                save(cfg)
                names.append(name)
            name = f'{label}_seed2026110940'
            cfg = paired(clone('control_huginn_zero_seed2026110931', name, **options),
                name, f'{label}_state32')
            cfg['phase'] = 'MATCHED_STATIC_CUT_SELECTION_NOT_ADAPTIVE_POLICY'
            save(cfg)
            names.append(name)
    write(ROOT/'state/static_granularity_grid.json', names)
    print(' '.join(names))


if __name__ == '__main__':
    main()
