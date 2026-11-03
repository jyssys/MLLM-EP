"""Retry the failed close and test fixed collector cost with a longer cohort.

The longer Huginn run is a diagnostic, not a relaxation of the gates.
All previous namespaces and results remain unchanged.
"""
import copy
from .common import *
from .prepare import put
from .run import execute
from .analysis_lane import quiet_analysis
from .verify_long_probes import verify
from .aligned_gates import one
from .ready_queue_audit import audit


def save_config(cfg, count, rate):
    split = 'resume_diagnostic'
    path = ROOT/'data'/f'{split}_{cfg["model"]}_L{cfg["length"]}.json'
    original = json.loads((ROOT/'data'/cfg['dataset']).read_text())
    rows = [copy.deepcopy(original[i % len(original)]) for i in range(count)]
    if path.exists():
        assert json.loads(path.read_text()) == rows
    else:
        write(path, rows)
    cfg['data_scope'] = dict(original_pool_records=len(original), offered_count=count,
        explicit_cyclic_reuse=count > len(original),
        scope='Existing public text prefixes. Repeated prompts are not independent dataset samples.')
    return put(cfg, count, rate, split=split)


def analyze_capture(cfg, clean):
    from ..looped_pd_headroom_recovery.profile_analysis import analyze
    name = cfg['name']
    with quiet_analysis(name+'_analysis'):
        profile = analyze(name, clean)
        gates = one(cfg)
        queue = audit(cfg, gates)
    correctness = verify(name)
    path = ROOT/'raw'/name/'D_probe_requests.jsonl'
    probes = [json.loads(s) for s in path.read_text().splitlines()] if path.exists() else []
    central = [p for p in probes if gates.get('observation_start', float('inf')) <= p['time']
        <= p['finish'] <= gates.get('observation_end', -1)]
    passed = bool(profile.get('eligible_for_matched_clean') and queue and queue.get('audited_P_gate')
        and gates.get('D_SLO_gate') and gates.get('D_credit_gate')
        and gates.get('D_credit_refusal_sample_fraction', 1) <= .5
        and central and correctness.get('exact') and correctness.get('full_work'))
    return dict(name=name, profile=profile, gates=gates, queue_audit=queue,
        correctness=correctness, central_probes=len(central), joint_screen_pass=passed,
        scope='Finite same-window operational screen. Not hardware-counter saturation or sustainable capacity.')


def main():
    activate()
    from ..looped_pd_atomic_prefill import evidence
    results = []
    # Change only the shutdown limit for the rejected Ouro capture.
    cfg = json.loads((ROOT/'configs/joint_confirm_ouro_s1_n128_nsys.json').read_text())
    cfg['name'] = 'resume_ouro_s1_n128_nsys_shutdown300'
    cfg['native_transport_limits'] = dict(cfg.get('native_transport_limits', {}), shutdown_timeout=300)
    p = ROOT/'configs'/(cfg['name']+'.json')
    if p.exists():
        assert json.loads(p.read_text()) == cfg
    else:
        write(p, cfg)
    write(ROOT/'state/resume_joint_review.json', dict(stage='OURO_SHUTDOWN_RETRY', time=time.time()))
    if execute(cfg['name']):
        results.append(analyze_capture(cfg, cfg['matched_clean']))
    else:
        results.append(dict(name=cfg['name'], status='FAILED_PROFILE_RETRY'))
    write(ROOT/'state/resume_joint_review_results.json', results)

    # Test the fixed-cost hypothesis. Both control and profile use 256 requests.
    # The capture window stays at three seconds. The original 128 results stand.
    cfg = json.loads((ROOT/'configs/joint_confirm_huginn_s1_n128_probe.json').read_text())
    cfg.update(name='resume_huginn_s1_n256_probe', timeout=1200, phase='final_repeat')
    cfg['native_transport_limits'] = dict(cfg.get('native_transport_limits', {}), shutdown_timeout=300)
    clean = save_config(cfg, 256, cfg['arrival_rate'])
    baseline = copy.deepcopy(cfg)
    baseline['name'] = 'resume_huginn_s1_n256_baseline'
    for key in ('D_probe_tokens', 'D_probe_count', 'D_probe_interval_s'):
        baseline.pop(key, None)
    bname = save_config(baseline, 256, cfg['arrival_rate'])
    write(ROOT/'state/resume_joint_review.json', dict(stage='HUGINN_MATCHED256', time=time.time()))
    if time.time() < LONG_START_CUTOFF and execute(bname) and execute(clean):
        parity = evidence.compare(bname, clean)
        correctness = verify(clean)
        if parity['passed'] and correctness.get('exact') and correctness.get('full_work'):
            prof = copy.deepcopy(cfg)
            prof.update(name='resume_huginn_s1_n256_nsys', matched_clean=clean,
                nsys=True, P_only_profiler=True, cuda_only_profile=False,
                bounded_native_capture_seconds=0., capture_role='prefill', capture_seconds=0.,
                autonomous_capture_seconds=3., capture_arrival_fraction=.35,
                capture_prewarmed=False, capture_until_close=False, profile_stop_at_exit=False,
                nsys_cuda_flush_ms=300000, profile_finalization_timeout=300)
            name = save_config(prof, 256, cfg['arrival_rate'])
            if execute(name):
                results.append(analyze_capture(prof, clean))
            else:
                results.append(dict(name=name, status='FAILED_PROFILE'))
        else:
            results.append(dict(name=clean, status='BLOCKED_CORRECTNESS', parity=parity, correctness=correctness))
    else:
        results.append(dict(name=clean, status='FAILED_OR_BUDGET_LIMITED'))
    write(ROOT/'state/resume_joint_review_results.json', results)
    write(ROOT/'state/resume_joint_review.json', dict(stage='COMPLETE_REVIEW_REQUIRED', time=time.time()))


if __name__ == '__main__':
    try:
        main()
    except BaseException as error:
        write(ROOT/'state/resume_joint_review.json', dict(stage='FAILED_REVIEW_REQUIRED', error=repr(error), time=time.time()))
        raise
