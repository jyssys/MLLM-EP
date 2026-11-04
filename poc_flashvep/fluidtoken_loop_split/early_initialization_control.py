"""Separate early CUPTI initialization from increased buffer preallocation."""
import copy
import time
from .common import *
from .run import execute
from .recovery_qualification import save


def main():
    initialize()
    write(ROOT / 'state/early_initialization_plan.json', dict(
        hypothesis='Early CUPTI initialization, rather than buffer count alone, prevents the late-attachment failure.',
        change='Retain early initialization and restore the documented three-buffer preallocation.',
        base='recovery_huginn_buffer16_window32_diagnostic512',
        inference_limit='A single successful repeat is mitigation evidence, not a complete native-driver root cause.'))
    pending = ROOT / 'state/admitted_buffer_recovery.json'
    while time.time() < LONG_START_CUTOFF:
        if pending.exists() and json.loads(pending.read_text())['status'] not in ('SCREENING', 'CONFIRMING'):
            break
        time.sleep(5)
    else:
        return
    cfg = copy.deepcopy(json.loads((ROOT / 'configs/recovery_huginn_buffer16_window32_diagnostic512.json').read_text()))
    cfg.update(name='recovery_huginn_early_buffer3_diagnostic512', cupti_preallocated_buffers=3)
    save(cfg)
    result = dict(name=cfg['name'], complete=execute(cfg['name']))
    if result['complete']:
        from ..looped_pd_autonomous_headroom.analysis_lane import quiet_analysis
        from .activity_analysis import analyze
        with quiet_analysis(cfg['name'] + '_parse'):
            result['profile'] = analyze(cfg['name'], cfg['matched_clean'])
    write(ROOT / 'state/early_initialization_result.json', result)


if __name__ == '__main__':
    main()
