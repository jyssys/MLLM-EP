"""Change only capture duration before independent longer-window repeats."""
from .common import *
from .run import execute
from .resume_joint_review import analyze_capture
from .independent_joint_repeats import captures


def main():
    activate()
    cfg = json.loads((ROOT/'configs/window_huginn_s1_nsys384.json').read_text())
    cfg.update(name='long_window_huginn_s1_nsys384', autonomous_capture_seconds=30.)
    path = ROOT/'configs'/(cfg['name']+'.json')
    assert not path.exists(), 'Preserve the previous namespace'
    write(path, cfg)
    write(ROOT/'state/huginn_long_window.json', dict(stage='DURATION_CONTROL', time=time.time()))
    row = dict(model='huginn', start=1, baseline='window_huginn_s1_baseline384',
        control='Only capture duration changes. See long_window_control_plan.md.')
    if execute(cfg['name']):
        row.update(analyze_capture(cfg, cfg['matched_clean']))
        row['measured_captures'] = captures(cfg['name'])
        row['accepted_joint_screen'] = row['joint_screen_pass'] and not row['measured_captures']
    else:
        row['status'] = 'PROFILE_FAILED'
    write(ROOT/'state/huginn_long_window_results.json', [row])
    if row.get('accepted_joint_screen') and time.time() < LONG_START_CUTOFF:
        from .huginn_window_validation import main as repeat
        repeat('long_window_huginn', 30., (2, 3), 'huginn_long_window_repeats')
    write(ROOT/'state/huginn_long_window.json', dict(stage='COMPLETE_REVIEW_REQUIRED', time=time.time()))


if __name__ == '__main__':
    try:
        main()
    except BaseException as error:
        write(ROOT/'state/huginn_long_window.json', dict(stage='FAILED_REVIEW_REQUIRED', error=repr(error), time=time.time()))
        raise
