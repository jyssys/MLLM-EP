"""Measure repeatability before attributing an intermittent stall to a cause."""
import copy
from .common import *
from .run import execute


def main():
    initialize()
    original = json.loads((ROOT / 'configs/recovery_huginn_original_profile384.json').read_text())
    results = []
    for label, seconds in (('repeat8', 8.), ('short3', 3.)):
        if __import__('time').time() >= LONG_START_CUTOFF:
            break
        cfg = copy.deepcopy(original)
        cfg.update(name='recovery_huginn_' + label + '_profile384',
                   autonomous_capture_seconds=seconds,
                   phase='PROFILER_REPRODUCIBILITY_CONTROL')
        path = ROOT / 'configs' / (cfg['name'] + '.json')
        if path.exists():
            assert json.loads(path.read_text()) == cfg
        else:
            write(path, cfg)
        complete = execute(cfg['name'])
        row = dict(name=cfg['name'], complete=complete)
        if complete:
            from ..looped_pd_autonomous_headroom.analysis_lane import quiet_analysis
            from ..looped_pd_headroom_recovery.profile_analysis import analyze
            with quiet_analysis(cfg['name'] + '_analysis'):
                row['profile'] = analyze(cfg['name'], cfg['matched_clean'])
        results.append(row)
        write(ROOT / 'state/recovery_capture_controls.json', results)
        # A new stall needs a native-stack diagnostic before another control.
        if not complete:
            break


if __name__ == '__main__':
    main()
