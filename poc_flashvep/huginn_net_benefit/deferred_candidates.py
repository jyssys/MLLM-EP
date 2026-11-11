"""Qualify delayed CODA delivery before any spatial-sharing performance claim."""
from .common import *
from .prepare import clone
from .optimization_cells import FAST, paired
from .concurrency_cells import options


def fields(arm, green):
    result = options(True) if arm == 'loop31' else dict(**FAST,
        independent_prompt_stream=True, independent_prompt_buffers=True,
        priority_decode_stream=True)
    result.update(deferred_coda_collection=True, independent_coda_buffers=True,
        deferred_coda_batches=2)
    if green:
        result['prompt_green_sms'] = 64
    return result


def main():
    initialize()
    names = []
    for green in (False, True):
        for arm in ('zero', 'loop31'):
            family = 'deferred_green64' if green else 'deferred'
            for count in (8, 32):
                name = f'{family}_{arm}_state{count}'
                cfg = clone(f'control_huginn_zero_state{count}', name,
                    phase='NONBLOCKING_FIXED_CODA_EXACTNESS_QUALIFICATION', **fields(arm, green))
                if count == 32:
                    cfg['required_qualification'] = f'{family}_{arm}_state8'
                save(cfg)
                names.append(name)
            name = f'{family}_{arm}_seed2026110940'
            cfg = paired(clone('control_huginn_zero_seed2026110931', name, **fields(arm, green)),
                name, f'{family}_{arm}_state32')
            cfg['phase'] = 'NONBLOCKING_CODA_SELECTION_NOT_CONFIRMED_ONLINE_GAIN'
            save(cfg)
            names.append(name)
    write(ROOT/'state/deferred_coda_jobs.json', names)
    print(' '.join(names))


if __name__ == '__main__':
    main()
