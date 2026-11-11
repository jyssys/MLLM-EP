"""Qualify no-checkpoint progressive Native transfer before performance use."""
from .common import *
from .prepare import clone


def main():
    initialize()
    names = []
    for count in (8, 32):
        name = f'candidate_native_earlytap_state{count}'
        cfg = clone(f'control_huginn_zero_state{count}', name,
            phase='NATIVE_ATOMIC_PROGRESSIVE_TRANSFER_EXACT_STATE',
            native_loop_event_transfer=True, early_transfer_depths=[16])
        if count == 32:
            cfg['required_qualification'] = 'candidate_native_earlytap_state8'
        save(cfg)
        names.append(name)
    write(ROOT / 'state/early_tap_jobs.json', names)
    print(' '.join(names))


if __name__ == '__main__':
    main()
