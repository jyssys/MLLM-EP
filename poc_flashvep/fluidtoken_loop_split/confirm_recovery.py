"""Run paired independent starts only after a qualified recovery screen."""
import copy
import random
import time
from .common import *
from .run import execute
from .recovery_analysis import analyze
from .recovery_qualification import save
from .state_comparison import compare as compare_states


def operational(screen):
    gates = screen['gates']
    return bool(screen['queue'] and screen['queue']['audited_P_gate']
                and gates['D_SLO_gate'] and gates['D_credit_gate']
                and gates['D_credit_refusal_sample_fraction'] <= .5
                and screen['central_completed_probes'] > 0)


def main():
    initialize()
    required = [ROOT / 'state/recovery_capture_controls.json',
                ROOT / 'state/recovery_qualification.json']
    while time.time() < LONG_START_CUTOFF:
        if all(path.exists() for path in required):
            controls, qualification = [json.loads(path.read_text()) for path in required]
            if len(controls) >= 2 and len(qualification) >= 4:
                break
        time.sleep(5)
    else:
        return
    states = compare_states('recovery_huginn_eager_state8', 'recovery_huginn_graph_state8')
    write(ROOT / 'state/recovery_prompt_states.json', states)
    if not states['exact'] or not qualification[-1]['probe_correctness']['exact']:
        write(ROOT / 'state/recovery_confirmation.json', dict(status='BLOCKED_CORRECTNESS'))
        return
    source = 'recovery_huginn_short3_profile384'
    screen = analyze(source)
    if not operational(screen):
        write(ROOT / 'state/recovery_confirmation.json', dict(status='SCREEN_FAILED_REVIEW_REQUIRED', screen=screen))
        return
    from ..looped_pd_atomic_prefill import evidence
    from ..looped_pd_autonomous_headroom import verify_long_probes
    verify_long_probes.execute = execute
    base = json.loads((ROOT / 'configs/recovery_huginn_original_clean384.json').read_text())
    results = []
    for start in (1, 2, 3):
        if time.time() >= LONG_START_CUTOFF:
            break
        cfg = copy.deepcopy(base)
        cfg.update(name=f'recovery_huginn_independent{start}_clean384',
                   arrival_seed=2026110400 + start, independent_start=start,
                   phase='INDEPENDENT_RECOVERY_CONFIRMATION')
        rng = random.Random(cfg['arrival_seed'])
        offsets = [0.]
        for _ in range(1, cfg['count']):
            offsets.append(offsets[-1] + rng.expovariate(cfg['arrival_rate']))
        cfg['arrival_offsets'] = offsets
        save(cfg)
        if not execute(cfg['name']):
            results.append(dict(start=start, status='CLEAN_FAILED'))
            break
        clean = cfg['name']
        profile = copy.deepcopy(cfg)
        profile.update(name=f'recovery_huginn_independent{start}_profile384',
                       matched_clean=clean, nsys=True, P_only_profiler=True,
                       autonomous_capture_seconds=3., capture_arrival_fraction=.35,
                       capture_role='prefill', capture_seconds=0.,
                       capture_prewarmed=False, capture_until_close=False,
                       profile_stop_at_exit=False, nsys_cuda_flush_ms=300000,
                       profile_finalization_timeout=300)
        save(profile)
        if not execute(profile['name']):
            results.append(dict(start=start, status='PROFILE_FAILED'))
            break
        from ..looped_pd_autonomous_headroom.analysis_lane import quiet_analysis
        from ..looped_pd_headroom_recovery.profile_analysis import analyze as parse
        with quiet_analysis(profile['name'] + '_parse'):
            parsed = parse(profile['name'], clean)
        screen = analyze(profile['name'])
        exact = verify_long_probes.verify(profile['name'])
        row = dict(start=start, clean=clean, profile=profile['name'],
                   trace=parsed, screen=screen, probe_correctness=exact,
                   passed=operational(screen) and exact['exact'] and exact['full_work'])
        results.append(row)
        write(ROOT / 'state/recovery_confirmation.json', dict(status='IN_PROGRESS', results=results))
        if not row['passed']:
            break
    write(ROOT / 'state/recovery_confirmation.json', dict(
        status='RECOVERY_THREE_START_GATE_PASSED' if len(results) == 3 and all(r.get('passed') for r in results)
        else 'CONFIRMATION_FAILED_OR_INCOMPLETE_REVIEW_REQUIRED', results=results,
        limitation='Short bounded captures. Native stall root cause remains inconclusive unless independently diagnosed.'))


if __name__ == '__main__':
    main()
