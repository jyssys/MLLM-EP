"""Qualify stream dependencies before any concurrency performance claim."""
from .common import *
from .prepare import clone


def main():
    initialize()
    names = []
    for count in (8, 32):
        name = f'candidate_streams_loop_state{count}'
        cfg = clone(f'secondary_huginn_loop_state{count}', name,
            phase='INDEPENDENT_STREAM_EXACT_DEPENDENCY_QUALIFICATION',
            independent_prompt_stream=True, priority_decode_stream=True,
            net_benefit_extension=True, selective_checkpoints=True,
            boundary_checkpoints=[16, 21, 24, 28, 30, 31])
        if count == 32:
            cfg['required_qualification'] = 'candidate_streams_loop_state8'
        save(cfg)
        names.append(name)
        split = dict(cfg, name=f'candidate_streams_buffers_loop_state{count}',
            independent_prompt_buffers=True)
        if count == 32:
            split['required_qualification'] = 'candidate_streams_buffers_loop_state8'
        save(split)
        names.append(split['name'])
    write(ROOT / 'state/independent_stream_jobs.json', names)
    print(' '.join(names))


if __name__ == '__main__':
    main()
