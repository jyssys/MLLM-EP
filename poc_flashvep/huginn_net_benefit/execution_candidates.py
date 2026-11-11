"""Prepare exact fixed-depth submission and final Decode-state diagnostics."""
from .common import *
from .prepare import clone


def main():
    initialize()
    jobs = []
    for count in (8, 32):
        for arm, options in (
                ('fixedsubmit', dict(fixed_depth_submit=True)),
                ('fixedsubmit_group32', dict(fixed_depth_submit=True, decode_loop_grouping=32))):
            name = f'candidate_{arm}_state{count}'
            cfg = clone(f'control_huginn_zero_state{count}', name,
                phase='EXACT_FIXED_DEPTH_SUBMISSION_QUALIFICATION', **options)
            if count == 32:
                cfg['required_qualification'] = f'candidate_{arm}_state8'
            save(cfg)
            jobs.append(name)
    for count in (8, 32):
        reference = f'qualification_finaldecode_zero_state{count}'
        for arm, options in (
                ('fixedsubmit', dict(fixed_depth_submit=True)),
                ('fixedsubmit_group32', dict(fixed_depth_submit=True, decode_loop_grouping=32))):
            name = f'qualification_finaldecode_{arm}_state{count}'
            cfg = clone(f'control_huginn_zero_state{count}', name,
                phase='EXACT_FINAL_DECODE_KV_NOT_CLEAN_TIMING',
                post_decode_state=True, post_decode_reference=reference, **options)
            cfg['required_qualification'] = f'candidate_{arm}_state{count}'
            save(cfg)
    write(ROOT / 'state/fixed_submission_jobs.json', jobs)
    print(' '.join(jobs))


if __name__ == '__main__':
    main()
