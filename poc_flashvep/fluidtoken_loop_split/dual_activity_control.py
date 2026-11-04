"""Measure both devices in one serving run after independent recovery."""
import copy
import time
from .common import *
from .run import execute
from .recovery_qualification import save


def main():
    initialize()
    target = ROOT / 'state/dual_activity_control.json'
    pending = ROOT / 'state/admitted_buffer_recovery.json'
    while time.time() < LONG_START_CUTOFF:
        if pending.exists():
            state = json.loads(pending.read_text())
            if state['status'] not in ('SCREENING', 'CONFIRMING'):
                break
        time.sleep(5)
    else:
        write(target, dict(status='NOT_RUN_TIME_BUDGET'))
        return
    if state['status'] != 'THREE_START_RECOVERY_PASSED':
        write(target, dict(status='BLOCKED_BY_HUGINN_GATE'))
        return
    cfg = copy.deepcopy(json.loads((ROOT / 'configs/recovery_huginn_admitted_buffer16_profile512.json').read_text()))
    cfg.update(name='recovery_huginn_dual_activity512', capture_roles=['prefill', 'decode'])
    save(cfg)
    if not execute(cfg['name']):
        write(target, dict(status='DUAL_CAPTURE_FAILED'))
        return
    from ..looped_pd_autonomous_headroom.analysis_lane import quiet_analysis
    from .activity_analysis import analyze
    from .recovery_analysis import analyze as screen
    from .confirm_activity_recovery import check
    with quiet_analysis(cfg['name'] + '_dual_parse'):
        profiles = {role: analyze(cfg['name'], cfg['matched_clean'], role)
                    for role in ('prefill', 'decode')}
    gates = screen(cfg['name'])
    result = check(cfg['name'], 'recovery_huginn_qualified_admission_no_probe512')
    write(target, dict(status='DUAL_SAME_RUN_PASS' if result['passed'] and
        all(p['eligible_for_matched_clean'] for p in profiles.values()) else 'DUAL_SCREEN_FAILED',
        profiles=profiles, qualification=result,
        scope='Both device traces belong to one execution. Gates use their time intersection. Kernel union is not SM occupancy.'))


if __name__ == '__main__':
    main()
