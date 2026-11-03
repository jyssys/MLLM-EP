"""Isolate collector activation from active NIXL traffic in one final control."""
from .common import *
from .run import execute
from .resume_joint_review import analyze_capture
from .independent_joint_repeats import captures
from .profiler_version_control import TOOL


def main():
    assert time.time() < LONG_START_CUTOFF
    assert TOOL.exists()
    os.environ['PATH'] = str(TOOL.parent)+os.pathsep+os.environ['PATH']
    cfg = json.loads((ROOT/'configs/nsys2025_short_huginn_s1_nsys384.json').read_text())
    clean = cfg['matched_clean']
    cfg.update(name='nsys2025_prewarm_huginn_s1_nsys384',
        autonomous_capture_seconds=0., capture_seconds=3., capture_prewarmed=True,
        capture_until_close=True, profile_stop_at_exit=True,
        diagnostic_scope='Collector starts before worker readiness and request warmup. No profiler start/stop occurs during active serving. Same clean source, arrivals, probes, and frozen gates.')
    write(ROOT/'configs'/(cfg['name']+'.json'), cfg)
    write(ROOT/'state/profiler_prewarm_control.json', dict(stage='PROFILE', name=cfg['name'], time=time.time()))
    row = dict(name=cfg['name'], model='huginn', clean=clean, status='PROFILE_FAILED')
    if execute(cfg['name']):
        row.update(analyze_capture(cfg, clean))
        row['measured_captures'] = captures(cfg['name'])
        row['accepted_joint_screen'] = bool(row['joint_screen_pass'] and not row['measured_captures'])
        row['status'] = 'COMPLETE_REVIEW_REQUIRED'
    write(ROOT/'state/profiler_prewarm_control_results.json', [row])
    write(ROOT/'state/profiler_prewarm_control.json', dict(stage='COMPLETE_REVIEW_REQUIRED', time=time.time()))


if __name__ == '__main__':
    try:
        main()
    except BaseException as error:
        write(ROOT/'state/profiler_prewarm_control.json', dict(stage='FAILED_REVIEW_REQUIRED', error=repr(error), time=time.time()))
        raise
