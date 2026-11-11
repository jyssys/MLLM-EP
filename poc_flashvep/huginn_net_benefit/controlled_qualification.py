"""Exercise real online admission in explicit, separated two-request bursts."""
from .common import *


def main():
    initialize()
    names = []
    for count in (8, 32):
        cfg = json.loads((ROOT/'configs'/f'candidate_fastloop_v2_state{count}.json').read_text())
        cfg.update(name=f'candidate_fastloop_controlled_state{count}',
            phase='CONTROLLED_BURST_ONLINE_ADMISSION_EXACTNESS_NOT_PRIMARY_SERVING',
            arrival_offsets=[10.*(index//2) for index in range(count)],
            arrival_pattern='TWO_REQUEST_BURSTS_SEPARATED_BY_10_SECONDS',
            qualification_scope='N requests in one online cohort. This is not N simultaneous submissions. The policy observes current D capacity; cuts are not assigned in advance.')
        if count == 32:
            cfg['required_qualification'] = 'candidate_fastloop_controlled_state8'
        else:
            cfg.pop('required_qualification', None)
        save(cfg)
        names.append(cfg['name'])
    cfg = json.loads((ROOT/'configs/selection_fastloop_v2_seed2026110940.json').read_text())
    cfg.update(name='selection_fastloop_controlqualified_seed2026110940',
        required_qualification='candidate_fastloop_controlled_state32',
        phase='PRIMARY_2048_16_SELECTION_AFTER_CONTROLLED_RUNTIME_QUALIFICATION')
    save(cfg)
    names.append(cfg['name'])
    write(ROOT/'state/controlled_online_qualification_jobs.json', names)
    print(' '.join(names))


if __name__ == '__main__':
    main()
