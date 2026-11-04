"""Require whole-run Decode SLO and dual-device evidence after tail recovery."""
import copy
import random
import time
from .common import *
from .run import execute
from .recovery_qualification import save


def pair(base, label, start, roles=('prefill', 'decode')):
    from ..looped_pd_autonomous_headroom.analysis_lane import quiet_analysis
    from ..looped_pd_atomic_prefill import evidence
    from .activity_analysis import analyze
    from .recovery_analysis import analyze as screen
    from .confirm_activity_recovery import check
    from .static_screen import metrics
    from .probe_reference import verify as quota_verify, verify_admitted
    verify = verify_admitted if base.get('D_probe_completion_contract') == 'admitted_exact_v2' else quota_verify
    cfg = copy.deepcopy(base)
    cfg.update(name=f'recovery_huginn_{label}_joint_s{start}_clean512',
        arrival_seed=2026111600 + start, independent_start=start,
        kernel_activity_seconds=0., capture_roles=list(roles),
        cupti_preallocated_buffers=16, allow_long_activity_diagnostic=True,
        observe_native_graph_coverage=True)
    rng = random.Random(cfg['arrival_seed'])
    cfg['arrival_offsets'] = [0.]
    for _ in range(1, cfg['count']):
        cfg['arrival_offsets'].append(cfg['arrival_offsets'][-1] + rng.expovariate(cfg['arrival_rate']))
    cfg.pop('matched_clean', None)
    save(cfg)
    clean = cfg['name']
    result = dict(label=label, start=start, clean=clean, passed=False)
    if not execute(clean):
        return dict(result, reason='CLEAN_EXECUTION_FAILED')
    with quiet_analysis(clean + '_whole_run_gate'):
        result['clean_metrics'] = metrics(clean)
        result['clean_output'] = evidence.compare('recovery_huginn_qualified_admission_no_probe512', clean)
        result['clean_probes'] = verify(clean)
    if not (result['clean_metrics']['Decode_SLO_screen'] and result['clean_output']['passed']
            and result['clean_probes']['exact'] and result['clean_probes']['full_work']):
        return dict(result, reason='CLEAN_WHOLE_RUN_GATE_FAILED')
    cfg.update(name=f'recovery_huginn_{label}_joint_s{start}_profile512',
        matched_clean=clean, kernel_activity_seconds=32., capture_arrival_fraction=.35)
    save(cfg)
    result['profile'] = cfg['name']
    if not execute(cfg['name']):
        return dict(result, reason='DUAL_CAPTURE_FAILED')
    with quiet_analysis(cfg['name'] + '_dual_activity_parse'):
        result['activity'] = {role: analyze(cfg['name'], clean, role) for role in roles}
        result['profile_metrics'] = metrics(cfg['name'])
    screen(cfg['name'])
    result['same_run'] = check(cfg['name'], 'recovery_huginn_qualified_admission_no_probe512')
    result['passed'] = bool(result['same_run']['passed']
        and result['profile_metrics']['Decode_SLO_screen']
        and all(row['eligible_for_matched_clean'] for row in result['activity'].values()))
    result['reason'] = 'ALL_GATES_PASS' if result['passed'] else 'DUAL_OR_WHOLE_RUN_GATE_FAILED'
    return result


def main():
    initialize()
    pending = ROOT / 'state/tail_geometry_control.json'
    target = ROOT / 'state/geometry_confirmation.json'
    write(ROOT / 'state/geometry_confirmation_plan.json', dict(
        order=['slots4_graph4', 'slots8_graph8'], starts=[1, 2, 3], count=512,
        rationale='Test the smaller qualified geometry first. Use the larger graph if the first geometry fails a joint gate.',
        requirements='Each clean and profiled run must pass whole-run Decode SLO. Each profile must pass same-window gates on both devices.',
        exclusions='No threshold change. No combination of evidence from different runs. One failed repeat prevents recovery readiness.'))
    while time.time() < LONG_START_CUTOFF:
        if pending.exists():
            state = json.loads(pending.read_text())
            if state['status'] == 'GEOMETRY_SCREEN_COMPLETE':
                break
        time.sleep(5)
    else:
        write(target, dict(status='NOT_RUN_TIME_BUDGET'))
        return
    candidates = [row for row in state['results'] if row.get('passed') and row['label'] != 'native8_graph4']
    results = []
    for candidate in candidates:
        base = json.loads((ROOT / 'configs' / (candidate['name'] + '.json')).read_text())
        trials = []
        for start in (1, 2, 3):
            if time.time() >= LONG_START_CUTOFF:
                break
            trial = pair(base, candidate['label'], start)
            trials.append(trial)
            write(target, dict(status='IN_PROGRESS', completed_candidates=results, current=trials))
            if not trial['passed']:
                break
        results.append(dict(candidate=candidate, trials=trials))
        if len(trials) == 3 and all(row['passed'] for row in trials):
            write(ROOT / 'state/huginn_final_recovery_gate.json', dict(
                status='RECOVERY_READY', same_run_gates=True, whole_run_decode_SLO=True,
                independent_repeats=True, exact_state_and_output=True,
                qualified_configuration=base, geometry_state_gate=candidate,
                results=trials, scope='Operational kernel-union and finite service gates. Not peak-FLOP saturation or useful migrated-work proof.'))
            write(target, dict(status='RECOVERY_READY', results=results))
            return
    write(target, dict(status='RECOVERY_FAILED', results=results))
    write(ROOT / 'state/huginn_final_recovery_gate.json', dict(
        status='RECOVERY_FAILED', results=results, scope='Retain bounded-window passes, but do not authorize method expansion.'))


if __name__ == '__main__':
    main()
