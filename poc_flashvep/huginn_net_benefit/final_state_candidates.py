"""Exclude startup requests from the stronger final Decode-state check."""
from .common import *
from .prepare import clone


def main():
    initialize()
    names = []
    for count in (8, 32):
        reference = f'qualification_finaldecode_v2_zero_state{count}'
        for arm, options in (('zero', {}), ('group32', dict(decode_loop_grouping=32)),
                ('fixedsubmit', dict(fixed_depth_submit=True)),
                ('fixedsubmit_group32', dict(fixed_depth_submit=True, decode_loop_grouping=32))):
            name = f'qualification_finaldecode_v2_{arm}_state{count}'
            cfg = clone(f'control_huginn_zero_state{count}', name,
                phase='FINAL_DECODE_STATE_MEASURED_REQUESTS_ONLY', post_decode_state=True,
                **options)
            if arm != 'zero':
                cfg['post_decode_reference'] = reference
            save(cfg)
            names.append(name)
    write(ROOT / 'state/final_decode_v2_jobs.json', names)
    print(' '.join(names))


if __name__ == '__main__':
    main()
