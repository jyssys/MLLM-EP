"""Test a longer predeclared Huginn window without changing acceptance limits."""
import copy
from .common import *
from .prepare import put
from .run import execute
from .analysis_lane import quiet_analysis
from .verify_long_probes import verify
from .resume_joint_review import analyze_capture
from .independent_joint_repeats import captures


def main(namespace='window_huginn', capture_seconds=8., start_numbers=(1, 2, 3), state_name='huginn_window_validation'):
    while time.time() < LONG_START_CUTOFF:
        state = ROOT/'state/resume_joint_review.json'
        if state.exists() and json.loads(state.read_text()).get('stage') in ('COMPLETE_REVIEW_REQUIRED', 'FAILED_REVIEW_REQUIRED'):
            break
        time.sleep(5)
    activate()
    from ..looped_pd_atomic_prefill import evidence, analyze as serving
    cfg0 = json.loads((ROOT/'configs/joint_confirm_huginn_s1_n128_probe.json').read_text())
    original = json.loads((ROOT/'data'/cfg0['dataset']).read_text())
    dataset = ROOT/'data/window_validation_huginn_L2560.json'
    rows = [copy.deepcopy(original[i % len(original)]) for i in range(384)]
    if dataset.exists():
        assert json.loads(dataset.read_text()) == rows
    else:
        write(dataset, rows)
    ledger = []
    for start in start_numbers:
        pair = (6, 7) if start == 2 else (4, 5)
        if time.time() >= LONG_START_CUTOFF:
            break
        cfg = copy.deepcopy(cfg0)
        cfg.update(name=f'{namespace}_s{start}_baseline384', warmup_count=3,
            arrival_seed=2026111000+start, independent_start=start, gpus=list(pair),
            phase='final_repeat', timeout=1500,
            data_scope=dict(original_pool_records=len(original), offered_count=384,
                explicit_cyclic_reuse=True, scope='Repeated public prefixes. Not 384 independent dataset samples.'))
        cfg['native_transport_limits'] = dict(cfg.get('native_transport_limits', {}), shutdown_timeout=300)
        for key in ('D_probe_tokens', 'D_probe_count', 'D_probe_interval_s'):
            cfg.pop(key, None)
        baseline = put(cfg, 384, cfg['arrival_rate'], split='window_validation')
        row = dict(model='huginn', start=start, baseline=baseline,
            diagnostic_contract=f'384 requests, {capture_seconds}-second capture, 24 local probes. No threshold change.')
        write(ROOT/'state'/(state_name+'.json'), dict(stage='BASELINE', name=baseline, time=time.time()))
        if not execute(baseline) or not evidence.full_work(baseline)['passed']:
            row['status'] = 'BLOCKED_BASELINE'
            ledger.append(row)
            break
        with quiet_analysis(baseline+'_metrics'):
            row['baseline_metrics'], _, _ = serving.serving_one(cfg)
        m = row['baseline_metrics']
        if m['request_TPOT_ms_p95'] > .9*m['TPOT_threshold_ms'] or m['token_ITL_ms_p99'] > m['ITL_threshold_ms']:
            row['status'] = 'BASELINE_D_SLO_FAILED'
            ledger.append(row)
            break
        probe = copy.deepcopy(cfg)
        probe.update(name=f'{namespace}_s{start}_probe384', D_probe_tokens=16,
            D_probe_count=24, D_probe_interval_s=2.)
        clean = put(probe, 384, cfg['arrival_rate'], split='window_validation')
        write(ROOT/'state'/(state_name+'.json'), dict(stage='PROBE', name=clean, time=time.time()))
        if not execute(clean) or not evidence.compare(baseline, clean)['passed']:
            row['status'] = 'BLOCKED_PROBE_OR_PARITY'
            ledger.append(row)
            break
        row['probe_correctness'] = verify(clean)
        if not all(row['probe_correctness'].get(k) for k in ('exact', 'full_work')):
            row['status'] = 'BLOCKED_PROBE_CORRECTNESS'
            ledger.append(row)
            break
        with quiet_analysis(clean+'_metrics'):
            row['probe_metrics'], _, _ = serving.serving_one(probe)
        prof = copy.deepcopy(probe)
        prof.update(name=f'{namespace}_s{start}_nsys384', matched_clean=clean,
            P_only_profiler=True, nsys=True, cuda_only_profile=False,
            bounded_native_capture_seconds=0., capture_role='prefill', capture_seconds=0.,
            autonomous_capture_seconds=capture_seconds, capture_arrival_fraction=.35,
            capture_prewarmed=False, capture_until_close=False, profile_stop_at_exit=False,
            nsys_cuda_flush_ms=300000, profile_finalization_timeout=300)
        name = put(prof, 384, cfg['arrival_rate'], split='window_validation')
        write(ROOT/'state'/(state_name+'.json'), dict(stage='PROFILE', name=name, time=time.time()))
        if execute(name):
            row.update(analyze_capture(prof, clean))
            row['measured_captures'] = {n: captures(n) for n in (baseline, clean, name)}
            row['fully_warmed'] = not any(row['measured_captures'].values())
            row['accepted_joint_screen'] = row['joint_screen_pass'] and row['fully_warmed']
        else:
            row['status'] = 'PROFILE_FAILED'
        ledger.append(row)
        write(ROOT/'state'/(state_name+'_results.json'), ledger)
        if not row.get('accepted_joint_screen'):
            break
    write(ROOT/'state'/(state_name+'_results.json'), ledger)
    write(ROOT/'state'/(state_name+'.json'), dict(stage='COMPLETE_REVIEW_REQUIRED', time=time.time()))


if __name__ == '__main__':
    try:
        main()
    except BaseException as error:
        write(ROOT/'state/huginn_window_validation.json', dict(stage='FAILED_REVIEW_REQUIRED', error=repr(error), time=time.time()))
        raise
