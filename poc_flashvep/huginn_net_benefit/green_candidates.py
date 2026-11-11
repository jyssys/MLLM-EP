"""Use documented resource controls, not guessed CUDA stream-structure offsets."""
from .common import *
from .prepare import clone
from .optimization_cells import FAST, paired
from .concurrency_cells import options


def main():
    initialize()
    jobs = []
    for arm in ('zero', 'loop31'):
        for count in (8, 32):
            name = f'green64_{arm}_state{count}'
            fields = (options(True) if arm == 'loop31' else dict(**FAST,
                independent_prompt_stream=True, independent_prompt_buffers=True,
                priority_decode_stream=True))
            cfg = clone(f'control_huginn_zero_state{count}', name,
                phase='DOCUMENTED_GREEN_CONTEXT_EXACTNESS_QUALIFICATION',
                prompt_green_sms=64, **fields)
            if count == 32:
                cfg['required_qualification'] = f'green64_{arm}_state8'
            save(cfg)
            jobs.append(name)
        name = f'green64_{arm}_seed2026110940'
        fields = (options(True) if arm == 'loop31' else dict(**FAST,
            independent_prompt_stream=True, independent_prompt_buffers=True,
            priority_decode_stream=True))
        cfg = paired(clone('control_huginn_zero_seed2026110931', name,
            prompt_green_sms=64, **fields), name, f'green64_{arm}_state32')
        cfg['phase'] = 'GREEN_CONTEXT_SELECTION_NOT_CONFIRMED_BENEFIT'
        save(cfg)
        jobs.append(name)
    write(ROOT/'state/green_context_qualification_jobs.json', jobs)
    print(' '.join(jobs))


if __name__ == '__main__':
    main()
