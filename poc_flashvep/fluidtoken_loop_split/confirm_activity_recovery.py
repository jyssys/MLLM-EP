"""Require fresh paired starts and exact work after an activity-only screen."""
import copy
import random
import time
from .common import *
from .run import execute
from .recovery_qualification import save
from .probe_reference import verify
from .confirm_recovery import operational


def check(name, baseline):
    from ..looped_pd_atomic_prefill import evidence
    from ..looped_pd_autonomous_headroom.strict_queue_audit import check as queue
    from ..looped_pd_autonomous_headroom.independent_joint_repeats import captures
    from ..looped_pd_autonomous_headroom.analysis_lane import quiet_analysis
    from .device_queue_audit import check as device_queue
    screen = json.loads((ROOT / 'raw' / name / 'recovery_joint_screen.json').read_text())
    with quiet_analysis(name + '_confirmation_checks'):
        strict = queue(name)
        device = device_queue(name)
        output = evidence.compare(baseline, name)
        cfg = json.loads((ROOT / 'configs' / (name + '.json')).read_text())
        if cfg.get('D_probe_completion_contract') == 'admitted_exact_v2':
            from .probe_reference import verify_admitted
            probes = verify_admitted(name)
        else:
            probes = verify(name)
        graph_captures = captures(name)
    return dict(name=name, screen=screen, event_only_queue=strict, device_qualified_queue=device, output_vs_no_probe=output,
        probes=probes, measured_model_graph_captures=graph_captures,
        passed=operational(screen) and device['queue_screen_pass'] and output['passed']
               and probes['exact'] and probes['full_work'] and not graph_captures)


def main():
    initialize()
    path = ROOT / 'state/activity_recovery_control.json'
    while time.time() < LONG_START_CUTOFF:
        if path.exists():
            initial = json.loads(path.read_text())
            break
        time.sleep(5)
    else:
        return
    if not initial.get('operational_pass'):
        write(ROOT / 'state/activity_recovery_confirmation.json', dict(status='SCREEN_FAILED_REVIEW_REQUIRED', initial=initial))
        return
    base = json.loads((ROOT / 'configs/recovery_huginn_extended1_clean512.json').read_text())
    baseline = copy.deepcopy(base)
    baseline['name'] = 'recovery_huginn_activity_no_probe512'
    for key in ('D_probe_tokens', 'D_probe_count', 'D_probe_interval_s'):
        baseline.pop(key, None)
    save(baseline)
    if not execute(baseline['name']):
        return
    ledger = [check(initial['name'], baseline['name'])]
    write(ROOT / 'state/activity_recovery_confirmation.json', dict(status='IN_PROGRESS', results=ledger))
    if not ledger[0]['passed']:
        return
    from ..looped_pd_autonomous_headroom.analysis_lane import quiet_analysis
    from .activity_analysis import analyze
    from .recovery_analysis import analyze as screen
    for start in (2, 3):
        if time.time() >= LONG_START_CUTOFF:
            break
        cfg = copy.deepcopy(base)
        cfg.update(name=f'recovery_huginn_activity_s{start}_clean512',
                   arrival_seed=2026111500 + start, independent_start=start)
        rng = random.Random(cfg['arrival_seed'])
        cfg['arrival_offsets'] = [0.]
        for _ in range(1, cfg['count']):
            cfg['arrival_offsets'].append(cfg['arrival_offsets'][-1] + rng.expovariate(cfg['arrival_rate']))
        save(cfg)
        if not execute(cfg['name']):
            ledger.append(dict(name=cfg['name'], passed=False, reason='CLEAN_FAILED'))
            break
        clean = cfg['name']
        cfg.update(name=f'recovery_huginn_activity_s{start}_profile512',
                   matched_clean=clean, kernel_activity_seconds=8., capture_arrival_fraction=.35,
                   capture_role='prefill', nsys=False, P_only_profiler=False,
                   autonomous_capture_seconds=0.)
        save(cfg)
        if not execute(cfg['name']):
            ledger.append(dict(name=cfg['name'], passed=False, reason='PROFILE_FAILED'))
            break
        with quiet_analysis(cfg['name'] + '_parse'):
            analyze(cfg['name'], clean)
        screen(cfg['name'])
        ledger.append(check(cfg['name'], baseline['name']))
        write(ROOT / 'state/activity_recovery_confirmation.json', dict(status='IN_PROGRESS', results=ledger))
        if not ledger[-1]['passed']:
            break
    passed = len(ledger) == 3 and all(row['passed'] for row in ledger)
    write(ROOT / 'state/activity_recovery_confirmation.json', dict(
        status='THREE_START_OPERATIONAL_GATE_PASSED' if passed else 'FAILED_OR_INCOMPLETE_REVIEW_REQUIRED',
        results=ledger, scope='Bounded same-run device windows. Not physical peak-FLOP saturation or sustainable serving capacity.'))


if __name__ == '__main__':
    main()
