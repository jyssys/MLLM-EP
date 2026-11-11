"""Prepare matched runtime ablations on a separate selection arrival trace."""
import random
from .common import *
from .prepare import clone


def main():
    initialize()
    arrival_seed = 2026110940
    rng = random.Random(arrival_seed)
    offsets = [0.]
    for _ in range(127):
        offsets.append(offsets[-1] + rng.expovariate(1.971))
    jobs = []
    for arm, options, qualification in (
            ('zero_a', {}, 'qualification_zero_state32'),
            ('group32', {'decode_loop_grouping': 32}, 'candidate_fixeddecode32_state32'),
            ('zero_b', {}, 'qualification_zero_state32')):
        name = f'selection_{arm}_seed{arrival_seed}'
        cfg = clone('control_huginn_zero_seed2026110931', name,
            phase='RUNTIME_SELECTION_NOT_HELD_OUT_CONFIRMATION',
            arrival_seed=arrival_seed, arrival_offsets=offsets,
            clean_confirmation=True, timeout=600,
            required_qualification=qualification, **options)
        save(cfg)
        jobs.append(name)
    write(ROOT / 'state/runtime_selection_jobs.json', jobs)
    print(' '.join(jobs))


if __name__ == '__main__':
    main()
