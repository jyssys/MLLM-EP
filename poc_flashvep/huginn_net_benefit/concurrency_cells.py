"""Forced diagnostic cuts separate execution feasibility from Online admission."""
from .common import *
from .prepare import clone
from .optimization_cells import FAST, paired


def options(concurrent):
    result = dict(**FAST, native_partial_cut=31, atomic_prompt=False,
        method='loop', axis='loop', cut=31, native_request_cuts=None,
        forced_offload_diagnostic=True)
    if concurrent:
        result.update(independent_prompt_stream=True, independent_prompt_buffers=True,
            priority_decode_stream=True)
    return result


def main():
    initialize()
    jobs = []
    for concurrent in (False, True):
        label = 'concurrent' if concurrent else 'serial'
        for count in (8, 32):
            name = f'forced_loop31_{label}_state{count}'
            cfg = clone(f'control_huginn_zero_state{count}', name,
                phase='FORCED_NATIVE_CUT_EXACTNESS_NOT_ONLINE_POLICY', **options(concurrent))
            if count == 32:
                cfg['required_qualification'] = f'forced_loop31_{label}_state8'
            save(cfg)
            jobs.append(name)
        name = f'forced_loop31_{label}_seed2026110940'
        cfg = paired(clone('control_huginn_zero_seed2026110931', name, **options(concurrent)),
            name, f'forced_loop31_{label}_state32')
        cfg['phase'] = 'FORCED_OFFLOAD_GPU_INTERVENTION_NOT_ONLINE_SAFETY_SUCCESS'
        save(cfg)
        jobs.append(name)
    write(ROOT/'state/forced_concurrency_jobs.json', jobs)
    print(' '.join(jobs))


if __name__ == '__main__':
    main()
