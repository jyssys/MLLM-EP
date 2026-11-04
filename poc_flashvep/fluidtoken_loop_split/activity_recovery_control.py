"""Try a reduced collector only if the NIXL setting does not qualify."""
import copy
import time
from .common import *
from .run import execute
from .recovery_qualification import save


def main():
    initialize()
    while time.time() < LONG_START_CUTOFF:
        test = ROOT / 'raw/activity_only_collector_selftest/activity_selftest.json'
        control = ROOT / 'state/manual_progress_control.json'
        if test.exists() and control.exists():
            tested = json.loads(test.read_text())
            result = json.loads(control.read_text())
            if 'profile_complete' in result:
                break
        time.sleep(5)
    else:
        return
    assert tested['passed'], 'Activity collector is not qualified'
    if result.get('operational_pass'):
        write(ROOT / 'state/activity_recovery_control.json', dict(status='NOT_REQUIRED_AFTER_OPERATIONAL_SCREEN'))
        return
    cfg = copy.deepcopy(json.loads((ROOT / 'configs/recovery_huginn_extended1_profile512.json').read_text()))
    cfg.update(name='recovery_huginn_activity_only_profile512', nsys=False,
               P_only_profiler=False, autonomous_capture_seconds=0.,
               kernel_activity_seconds=8., phase='ACTIVITY_ONLY_COLLECTOR_CONTROL',
               matched_clean='recovery_huginn_extended1_clean512')
    save(cfg)
    complete = execute(cfg['name'])
    result = dict(name=cfg['name'], complete=complete)
    if complete:
        from ..looped_pd_autonomous_headroom.analysis_lane import quiet_analysis
        from .activity_analysis import analyze
        with quiet_analysis(cfg['name'] + '_parse'):
            result['profile'] = analyze(cfg['name'], cfg['matched_clean'])
        from .recovery_analysis import analyze as screen
        from .confirm_recovery import operational
        result['screen'] = screen(cfg['name'])
        result['operational_pass'] = operational(result['screen'])
    write(ROOT / 'state/activity_recovery_control.json', result)


if __name__ == '__main__':
    main()
