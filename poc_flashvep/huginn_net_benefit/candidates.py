"""Create small ablations before confirmation or combination."""
import argparse
from .common import *
from .prepare import clone


def main():
    p = argparse.ArgumentParser()
    p.add_argument('kind', choices=('async', 'selective', 'events', 'python'))
    kind = p.parse_args().kind
    initialize()
    jobs = []
    if kind in ('async', 'selective'):
        for count in (8, 32):
            name = f'candidate_{kind}_state{count}'
            source = (f'control_huginn_zero_state{count}' if kind == 'async'
                else f'secondary_huginn_loop_state{count}')
            cfg = clone(source, name, phase='CANDIDATE_EXACT_STATE_NOT_PERFORMANCE')
            if kind == 'async':
                cfg['asynchronous'] = True
            else:
                cfg.update(net_benefit_extension=True, selective_checkpoints=True,
                    boundary_checkpoints=[16, 21, 24, 28, 30, 31])
            save(cfg)
            jobs.append(name)
    elif kind == 'python':
        name = 'cost_zero_python_n8'
        cfg = clone('control_huginn_zero_state8', name,
            phase='DIAGNOSTIC_PYTHON_ATTRIBUTION_NOT_PERFORMANCE',
            state_probe=False, async_prompt_state_probe=False,
            clean_confirmation=False, profile_python=True, host_cost_trace=True)
        save(cfg)
        jobs.append(name)
    else:
        for arm, source in (('zero', 'control_huginn_zero_seed2026110931'),
                ('checkpoint', 'recoveryr_huginn_loop_seed2026110931'),
                ('loop', 'recoveryr_huginn_loop_seed2026110931')):
            name = f'cost_{arm}_events_seed2026110931'
            cfg = clone(source, name, phase='COST_INSTRUMENTED_SEPARATE_CONTROL',
                clean_confirmation=False, sparse_device_trace=True, sparse_decode_period=97,
                host_cost_trace=True, observe_transport=True, observe_descriptor_cost=True)
            if arm == 'checkpoint':
                cfg.update(max_continuations=0, force_destination_reject=True)
            save(cfg)
            jobs.append(name)
    write(ROOT / 'state' / f'{kind}_jobs.json', jobs)
    print(' '.join(jobs))


if __name__ == '__main__':
    main()
