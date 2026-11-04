"""Evaluate bounded changes after the longer-window diagnostic."""
import copy
import random
import time
from .common import *
from .run import execute
from .recovery_qualification import save
from .confirm_recovery import operational


def paired(cfg, prefix, seed=None):
    cfg = copy.deepcopy(cfg)
    if seed is not None:
        cfg['arrival_seed'] = seed
        rng = random.Random(seed)
        cfg['arrival_offsets'] = [0.]
        for _ in range(1, cfg['count']):
            cfg['arrival_offsets'].append(cfg['arrival_offsets'][-1] + rng.expovariate(cfg['arrival_rate']))
    cfg.update(name=prefix + '_clean512', kernel_activity_seconds=0., nsys=False,
               P_only_profiler=False, autonomous_capture_seconds=0.)
    cfg.pop('matched_clean', None)
    save(cfg)
    clean = cfg['name']
    if not execute(clean):
        return dict(name=clean, complete=False, passed=False)
    cfg.update(name=prefix + '_profile512', matched_clean=clean,
               kernel_activity_seconds=32., capture_arrival_fraction=.35)
    save(cfg)
    if not execute(cfg['name']):
        return dict(name=cfg['name'], complete=False, passed=False)
    from ..looped_pd_autonomous_headroom.analysis_lane import quiet_analysis
    from ..looped_pd_autonomous_headroom.strict_queue_audit import check
    from .activity_analysis import analyze
    from .recovery_analysis import analyze as screen
    from .probe_reference import verify
    from .tail_attribution import analyze as tails
    from .device_queue_audit import check as device_queue
    with quiet_analysis(cfg['name'] + '_profile_parse'):
        profile = analyze(cfg['name'], clean)
    gates = screen(cfg['name'])
    with quiet_analysis(cfg['name'] + '_exact_and_queue'):
        queue = check(cfg['name'])
        device = device_queue(cfg['name'])
        exact = verify(cfg['name'])
        tail = tails(cfg['name'])
    return dict(name=cfg['name'], complete=True, profile=profile, screen=gates,
        event_only_queue=queue, device_qualified_queue=device, exact_probe=exact, tail_overlap=tail,
        passed=operational(gates) and device['queue_screen_pass'] and exact['exact'] and exact['full_work'])


def main():
    initialize()
    path = ROOT / 'state/long_activity_result.json'
    while not path.exists() and time.time() < LONG_START_CUTOFF:
        time.sleep(5)
    if not path.exists():
        return
    initial = json.loads(path.read_text())
    if initial.get('operational_pass'):
        write(ROOT / 'state/adaptive_activity_recovery.json', dict(status='LONG_WINDOW_SCREEN_PASSED_REQUIRES_CONFIRMATION'))
        return
    cfg = json.loads((ROOT / 'configs/recovery_huginn_activity_window32_profile512.json').read_text())
    cfg['D_probe_max_active_decode'] = 2
    write(ROOT / 'state/adaptive_activity_plan.json', dict(
        first='Defer independent D-local probes when more than two background recurrent requests are ready.',
        reason='The first activity capture passed GPU union but failed same-window ITL. Admission uses current state only.',
        second='If queue pressure still fails, test QPS 2.2 with the same admission rule and frozen SLO.',
        unchanged='Original weights, BF16, R32, native NIXL, model graphs, full work, 32-second window at arrival fraction 0.35.',
        exclusions='No background request is dropped, deferred by this hook, or shortened. No successful probe means no headroom claim.'))
    results = []
    for label, rate in (('admitted_probe', 1.971), ('admitted_probe_q22', 2.2)):
        if time.time() >= LONG_START_CUTOFF:
            break
        changed_rate = rate != cfg['arrival_rate']
        cfg['arrival_rate'] = rate
        result = paired(cfg, 'recovery_huginn_' + label,
                        cfg['arrival_seed'] if changed_rate else None)
        results.append(result)
        write(ROOT / 'state/adaptive_activity_recovery.json', dict(status='SCREENING', results=results))
        if result['passed']:
            break
        if not result['complete']:
            break
    write(ROOT / 'state/adaptive_activity_recovery.json', dict(
        status='SCREEN_PASSED_REQUIRES_CONFIRMATION' if results and results[-1]['passed'] else 'REVIEW_REQUIRED',
        results=results))


if __name__ == '__main__':
    main()
