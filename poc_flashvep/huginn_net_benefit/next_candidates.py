"""Prepare the next isolated qualification and selection cells."""
from .common import *
from .prepare import clone


def main():
    initialize()
    for count in (8, 32):
        cfg = clone(f'control_huginn_zero_state{count}',
            f'candidate_integer_validation_state{count}',
            phase='EXACT_INTEGER_VALIDATION_FAST_PATH', exact_integer_validation=True)
        if count == 32:
            cfg['required_qualification'] = 'candidate_integer_validation_state8'
        save(cfg)
    cfg = json.loads((ROOT / 'configs/selection_zero_a_seed2026110940.json').read_text())
    save(dict(cfg, name='selection_zero_c_seed2026110940'))
    jobs = [
        'qualification_finaldecode_v2_zero_state8',
        'qualification_finaldecode_v2_group32_state8',
        'qualification_finaldecode_v2_fixedsubmit_state8',
        'qualification_finaldecode_v2_fixedsubmit_group32_state8',
        'candidate_blockwire_state8', 'candidate_native_earlytap_state8',
        'candidate_streams_loop_state8',
        'selection_fixedsubmit_seed2026110940',
        'selection_fixedsubmit_group32_seed2026110940',
        'selection_zero_c_seed2026110940',
        'candidate_blockwire_state32', 'candidate_native_earlytap_state32',
        'candidate_streams_loop_state32', 'candidate_wire256_state32',
        'candidate_wire512_state8', 'candidate_integer_validation_state8',
        'candidate_integer_validation_state32']
    write(ROOT / 'state/next_qualification_selection_jobs.json', jobs)
    print(' '.join(jobs))


if __name__ == '__main__':
    main()
