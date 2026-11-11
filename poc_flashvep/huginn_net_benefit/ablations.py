"""Predeclare matched no-migration and current-policy ablations."""
import argparse
from .common import *
from .prepare import clone


def main():
    p = argparse.ArgumentParser()
    p.add_argument('stage', choices=('state', 'serving'))
    stage = p.parse_args().stage
    initialize()
    jobs = []
    if stage == 'state':
        for count in (8, 32):
            for arm in ('zero', 'checkpoint'):
                source = (f'control_huginn_zero_state{count}' if arm == 'zero'
                    else f'secondary_huginn_loop_state{count}')
                name = f'qualification_{arm}_state{count}'
                cfg = clone(source, name, phase='FRESH_CHECKPOINT_ABLATION_EXACT_STATE',
                    require_actual_offload_qualification=False)
                if arm == 'checkpoint':
                    cfg.update(boundary_checkpoints=[16, 21, 24, 28, 30, 31], max_continuations=0,
                        control_arm='All checkpoints, no migration', force_destination_reject=True)
                save(cfg)
                jobs.append(name)
    else:
        seed = 2026110931
        sources = dict(native=f'recovery_huginn_native_seed{seed}',
            checkpoint=f'recoveryr_huginn_loop_seed{seed}',
            previous=f'recoveryr_huginn_loop_seed{seed}', zero=f'control_huginn_zero_seed{seed}')
        for arm in ('native', 'checkpoint', 'previous', 'zero'):
            name = f'ablation_{arm}_seed{seed}'
            cfg = clone(sources[arm], name, phase='MATCHED_CHECKPOINT_COST_ABLATION', timeout=600)
            if arm == 'checkpoint':
                cfg.update(max_continuations=0, force_destination_reject=True,
                    control_arm='All checkpoints, no migration')
            save(cfg)
            jobs.append(name)
    write(ROOT / 'state' / f'ablation_{stage}_jobs.json', jobs)
    print(' '.join(jobs))


if __name__ == '__main__':
    main()
