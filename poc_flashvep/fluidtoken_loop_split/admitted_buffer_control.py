"""Combine only independently tested collector and probe-admission controls."""
import copy
import time
from .common import *
from .run import execute
from .recovery_qualification import save


def main():
    initialize()
    path = ROOT / 'state/buffer_recovery_result_v2.json'
    while time.time() < LONG_START_CUTOFF:
        if path.exists() and 'complete' in json.loads(path.read_text()):
            result = json.loads(path.read_text())
            break
        time.sleep(5)
    else:
        return
    target = ROOT / 'state/admitted_buffer_recovery.json'
    if not result['complete']:
        write(target, dict(status='BLOCKED_COLLECTOR_RECOVERY'))
        return
    cfg = json.loads((ROOT / 'configs/recovery_huginn_admitted_probe_clean512.json').read_text())
    cfg.update(name='recovery_huginn_admitted_buffer16_profile512',
               matched_clean='recovery_huginn_admitted_probe_clean512',
               kernel_activity_seconds=32., capture_arrival_fraction=.35,
               cupti_preallocated_buffers=16, allow_long_activity_diagnostic=True)
    save(cfg)
    if not execute(cfg['name']):
        write(target, dict(status='PROFILE_FAILED'))
        return
    from ..looped_pd_autonomous_headroom.analysis_lane import quiet_analysis
    from .activity_analysis import analyze
    from .recovery_analysis import analyze as screen
    from .confirm_recovery import operational
    from .tail_attribution import analyze as tails
    with quiet_analysis(cfg['name'] + '_parse'):
        trace = analyze(cfg['name'], cfg['matched_clean'])
    gates = screen(cfg['name'])
    with quiet_analysis(cfg['name'] + '_tail'):
        tails(cfg['name'])
    results = [dict(name=cfg['name'], profile=trace, screen=gates, passed=operational(gates))]
    write(target, dict(status='SCREENING', results=results))
    if not results[-1]['passed']:
        # A second offered rate is a discovery control, not independent confirmation.
        from .adaptive_activity_recovery import paired
        cfg['arrival_rate'] = 2.2
        row = paired(cfg, 'recovery_huginn_buffer16_admitted_q22', cfg['arrival_seed'])
        results.append(row)
        write(target, dict(status='SCREENING', results=results))
        if not row['passed']:
            write(target, dict(status='SCREEN_FAILED', results=results))
            return
        cfg = json.loads((ROOT / 'configs' / (row['name'] + '.json')).read_text())
    baseline = copy.deepcopy(cfg)
    baseline.update(name='recovery_huginn_qualified_admission_no_probe512', kernel_activity_seconds=0.)
    for key in ('D_probe_tokens', 'D_probe_count', 'D_probe_interval_s', 'matched_clean'):
        baseline.pop(key, None)
    save(baseline)
    if not execute(baseline['name']):
        write(target, dict(status='NO_PROBE_BASELINE_FAILED', results=results))
        return
    from .confirm_activity_recovery import check
    from .adaptive_activity_recovery import paired
    qualification = [check(cfg['name'], baseline['name'])]
    for start in (2, 3):
        if not qualification[-1]['passed'] or time.time() >= LONG_START_CUTOFF:
            break
        row = paired(cfg, f'recovery_huginn_buffer16_confirm{start}', 2026111600 + start)
        results.append(row)
        if not row['passed']:
            qualification.append(dict(name=row['name'], passed=False, reason='INDEPENDENT_SCREEN_FAILED'))
            break
        qualification.append(check(row['name'], baseline['name']))
        write(target, dict(status='CONFIRMING', results=results, qualification=qualification))
    passed = len(qualification) == 3 and all(row['passed'] for row in qualification)
    write(target, dict(status='THREE_START_RECOVERY_PASSED' if passed else 'CONFIRMATION_FAILED',
                       results=results, qualification=qualification))


if __name__ == '__main__':
    main()
