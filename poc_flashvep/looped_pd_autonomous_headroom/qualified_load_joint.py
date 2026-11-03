"""Check the strongest accepted finite probe load in one P capture."""
from .common import *
from .run import execute
from .resume_joint_review import analyze_capture
from .independent_joint_repeats import captures


def main():
    state = ROOT/'state/qualified_probe_load.json'
    while time.time() < LONG_START_CUTOFF:
        if state.exists() and json.loads(state.read_text()).get('stage') in (
                'COMPLETE_REVIEW_REQUIRED', 'FAILED_REVIEW_REQUIRED', 'NO_ACCEPTED_SOURCE'):
            break
        time.sleep(5)
    records = ROOT/'state/qualified_probe_load_results.json'
    candidates = [r for r in json.loads(records.read_text()) if r.get('accepted')] if records.exists() else []
    if not candidates or time.time() >= LONG_START_CUTOFF:
        write(ROOT/'state/qualified_load_joint.json', dict(stage='NOT_RUN_NO_ACCEPTED_SOURCE_OR_BUDGET', time=time.time()))
        return
    selected = max(candidates, key=lambda row: row['additional_prompt_tokens_per_response_wall_s'])
    cfg = json.loads((ROOT/'configs'/(selected['name']+'.json')).read_text())
    cfg.update(name='qualified_load_joint_ouro128', matched_clean=selected['name'],
        P_only_profiler=True, nsys=True, capture_role='prefill', capture_seconds=3.,
        capture_prewarmed=True, capture_until_close=True, profile_stop_at_exit=True,
        nsys_cuda_flush_ms=300000, profile_finalization_timeout=300)
    write(ROOT/'configs'/(cfg['name']+'.json'), cfg)
    write(ROOT/'state/qualified_load_joint.json', dict(stage='PROFILE', name=cfg['name'], time=time.time()))
    row = dict(name=cfg['name'], selected_clean=selected['name'], status='PROFILE_FAILED')
    if execute(cfg['name']):
        row.update(analyze_capture(cfg, selected['name']))
        row['measured_captures'] = captures(cfg['name'])
        row['accepted_joint_screen'] = bool(row['joint_screen_pass'] and not row['measured_captures'])
        row['status'] = 'COMPLETE_REVIEW_REQUIRED'
    write(ROOT/'state/qualified_load_joint_results.json', [row])
    write(ROOT/'state/qualified_load_joint.json', dict(stage='COMPLETE_REVIEW_REQUIRED', time=time.time()))


if __name__ == '__main__':
    try:
        main()
    except BaseException as error:
        write(ROOT/'state/qualified_load_joint.json', dict(stage='FAILED_REVIEW_REQUIRED', error=repr(error), time=time.time()))
        raise
