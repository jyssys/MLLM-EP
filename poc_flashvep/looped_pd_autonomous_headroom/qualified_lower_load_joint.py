"""Test one lower probe load after the highest load fails the central ITL gate."""
from .common import *
from .run import execute
from .resume_joint_review import analyze_capture
from .independent_joint_repeats import captures


def main():
    assert time.time() < LONG_START_CUTOFF
    cfg = json.loads((ROOT/'configs/load_ouro_tokens128_500ms_probe128.json').read_text())
    clean = cfg['name']
    cfg.update(name='qualified_lower_load_joint_ouro128', matched_clean=clean,
        P_only_profiler=True, nsys=True, capture_role='prefill', capture_seconds=3.,
        capture_prewarmed=True, capture_until_close=True, profile_stop_at_exit=True,
        nsys_cuda_flush_ms=300000, profile_finalization_timeout=300)
    write(ROOT/'configs'/(cfg['name']+'.json'), cfg)
    write(ROOT/'state/qualified_lower_load_joint.json', dict(stage='PROFILE', name=cfg['name'], time=time.time()))
    row = dict(name=cfg['name'], selected_clean=clean, status='PROFILE_FAILED')
    if execute(cfg['name']):
        row.update(analyze_capture(cfg, clean))
        row['measured_captures'] = captures(cfg['name'])
        row['accepted_joint_screen'] = bool(row['joint_screen_pass'] and not row['measured_captures'])
        row['status'] = 'COMPLETE_REVIEW_REQUIRED'
    write(ROOT/'state/qualified_lower_load_joint_results.json', [row])
    write(ROOT/'state/qualified_lower_load_joint.json', dict(stage='COMPLETE_REVIEW_REQUIRED', time=time.time()))


if __name__ == '__main__':
    try:
        main()
    except BaseException as error:
        write(ROOT/'state/qualified_lower_load_joint.json', dict(stage='FAILED_REVIEW_REQUIRED', error=repr(error), time=time.time()))
        raise
