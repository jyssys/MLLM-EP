"""Test a predeclared longer window without changing the serving workload."""
import copy
from .common import *
from .run import execute
from .recovery_qualification import save


def main():
    initialize()
    cfg = copy.deepcopy(json.loads((ROOT / 'configs/recovery_huginn_activity_only_profile512.json').read_text()))
    cfg.update(name='recovery_huginn_activity_window32_profile512',
               kernel_activity_seconds=32., capture_arrival_fraction=.35,
               phase='PREDECLARED_LONGER_ACTIVITY_WINDOW')
    write(ROOT / 'state/long_activity_plan.json', dict(
        hypothesis='An eight-second interval is insufficient to characterize sustained queue pressure.',
        change='Capture duration only: 8 to 32 seconds. Start fraction remains 0.35.',
        limitations='The failed eight-second result remains a failure. No post-hoc window selection.',
        config=cfg['name']))
    save(cfg)
    result = dict(name=cfg['name'], complete=execute(cfg['name']))
    if result['complete']:
        from ..looped_pd_autonomous_headroom.analysis_lane import quiet_analysis
        from .activity_analysis import analyze
        from .recovery_analysis import analyze as screen
        from .confirm_recovery import operational
        with quiet_analysis(cfg['name'] + '_parse'):
            result['profile'] = analyze(cfg['name'], cfg['matched_clean'])
        result['screen'] = screen(cfg['name'])
        result['operational_pass'] = operational(result['screen'])
    write(ROOT / 'state/long_activity_result.json', result)


if __name__ == '__main__':
    main()
