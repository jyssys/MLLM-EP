"""Collect optional same-run D activity after the required recovery repeats."""
import copy
import time
from .common import *
from .run import execute
from .recovery_qualification import save


def main():
    initialize()
    target = ROOT / 'state/dual_device_paced_recovery_control.json'
    pending = ROOT / 'state/paced_recovery_v5.json'
    while time.time() < LONG_START_CUTOFF:
        if pending.exists():
            state = json.loads(pending.read_text())
            if state['status'] != 'IN_PROGRESS':
                break
        time.sleep(5)
    else:
        write(target, dict(status='NOT_RUN_TIME_BUDGET'))
        return
    if state['status'] != 'RECOVERY_READY':
        write(target, dict(status='BLOCKED_BY_HUGINN_GATE'))
        return
    from .conditional_split_qualification import recovery_ready
    assert recovery_ready(json.loads((ROOT / 'state/huginn_final_recovery_gate.json').read_text()))
    reference = state['results'][0]
    cfg = copy.deepcopy(json.loads((ROOT / 'configs' / (reference['profile'] + '.json')).read_text()))
    cfg.update(name='recovery_huginn_paced8_v5_dual_activity512', capture_roles=['prefill', 'decode'])
    save(cfg)
    if not execute(cfg['name']):
        write(target, dict(status='CAPTURE_FAILED', name=cfg['name']))
        return
    from ..looped_pd_autonomous_headroom.analysis_lane import quiet_analysis
    from .activity_analysis import analyze
    from .recovery_analysis import analyze as screen
    from .confirm_activity_recovery import check
    from .static_screen import metrics
    with quiet_analysis(cfg['name'] + '_dual_activity_analysis'):
        activity = {role: analyze(cfg['name'], reference['clean'], role)
                    for role in ('prefill', 'decode')}
        whole = metrics(cfg['name'])
    screen(cfg['name'])
    same = check(cfg['name'], 'recovery_huginn_qualified_admission_no_probe512')
    passed = (same['passed'] and whole['Decode_SLO_screen']
              and all(row['eligible_for_matched_clean'] for row in activity.values()))
    write(target, dict(status='SUPPLEMENTAL_DUAL_CAPTURE_PASS' if passed else 'SUPPLEMENTAL_CAPTURE_REJECTED',
          name=cfg['name'], matched_clean=reference['clean'], activity=activity,
          same_run=same, whole_run=whole,
          scope='One optional paired dual-device capture. It does not replace any required independent recovery start.'))


if __name__ == '__main__':
    main()
