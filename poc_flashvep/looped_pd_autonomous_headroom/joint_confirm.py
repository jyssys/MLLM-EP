"""Confirm exact D graph probes under capacity-derived Prefill pressure."""
import copy
from .common import *
from .prepare import put
from .run import execute
from .verify_long_probes import verify
from .analysis_lane import quiet_analysis
from .aligned_gates import one
from .ready_queue_audit import audit


def main():
    activate()
    from ..looped_pd_atomic_prefill import evidence, analyze as serving
    from ..looped_pd_headroom_recovery.profile_analysis import analyze
    sources = json.loads((ROOT/'state/d_graph_state_followup_results.json').read_text())
    ledger = []
    for model in ('ouro', 'huginn'):
        chosen = next((r for r in sources if r['model'] == model), {})
        if not (chosen.get('background_parity', {}).get('passed') and
                chosen.get('probe_correctness', {}).get('exact') and
                chosen.get('probe_correctness', {}).get('full_work') and
                chosen.get('state_correctness', {}).get('state_exact')):
            ledger.append(dict(model=model, status='BLOCKED_SOURCE_CORRECTNESS'))
            continue
        original = json.loads((ROOT/'configs'/(chosen['name']+'.json')).read_text())
        capacity = ROOT/'raw'/f'capacity_bulk_written_{model}_matched'/'cohorts.jsonl'
        rates = [r['job']['batch']/r['prefill_host_service_s'] for r in map(json.loads, capacity.read_text().splitlines()) if r['status'] == 'COMPLETE']
        rate = round(1.3*max(rates), 3)
        passed = False
        for start in (1, 2, 3):
            if time.time() >= LONG_START_CUTOFF or (start > 1 and not passed):
                break
            passed = False
            for count in (128, 256):
                if time.time() >= LONG_START_CUTOFF:
                    break
                cfg = copy.deepcopy(original)
                cfg.update(name=f'joint_confirm_{model}_s{start}_n{count}_baseline',
                    phase='final_repeat', arrival_seed=2026110800+start,
                    independent_start=start, timeout=1200,
                    gpus=list(((0, 1), (2, 3), (0, 1))[start-1] if model == 'ouro'
                              else ((4, 5), (6, 7), (4, 5))[start-1]))
                for key in ('D_probe_tokens', 'D_probe_count', 'D_probe_interval_s'):
                    cfg.pop(key, None)
                baseline = put(cfg, count, rate, split='confirmation')
                write(ROOT/'state/joint_confirm.json', dict(stage='BASELINE', name=baseline, time=time.time()))
                row = dict(model=model, start=start, count=count, QPS=rate, baseline=baseline,
                    pressure_rule='1.3 times the strongest matched finite P-only rate. Fixed before this confirmation set.')
                if not execute(baseline) or not evidence.full_work(baseline)['passed']:
                    row['status'] = 'BLOCKED_BASELINE'
                    ledger.append(row)
                    break
                with quiet_analysis(baseline+'_metrics'):
                    metrics, _, _ = serving.serving_one(cfg)
                row['baseline_metrics'] = metrics
                if metrics['request_TPOT_ms_p95'] > .9*metrics['TPOT_threshold_ms'] or metrics['token_ITL_ms_p99'] > metrics['ITL_threshold_ms']:
                    row['status'] = 'BASELINE_D_SLO_FAILED'
                    ledger.append(row)
                    break
                probe = copy.deepcopy(cfg)
                probe.update(name=f'joint_confirm_{model}_s{start}_n{count}_probe',
                    D_probe_tokens=16, D_probe_count=8, D_probe_interval_s=2.)
                clean = put(probe, count, rate, split='confirmation')
                write(ROOT/'state/joint_confirm.json', dict(stage='PROBE', name=clean, time=time.time()))
                if not execute(clean) or not evidence.compare(baseline, clean)['passed']:
                    row['status'] = 'BLOCKED_PROBE_EXECUTION_OR_PARITY'
                    ledger.append(row)
                    break
                row['clean_probe_correctness'] = verify(clean)
                if not all(row['clean_probe_correctness'].get(k) for k in ('exact', 'full_work')):
                    row['status'] = 'BLOCKED_PROBE_REFERENCE'
                    ledger.append(row)
                    break
                prof = copy.deepcopy(probe)
                prof.update(name=f'joint_confirm_{model}_s{start}_n{count}_nsys',
                    matched_clean=clean, P_only_profiler=True, nsys=True,
                    cuda_only_profile=False, bounded_native_capture_seconds=0.,
                    capture_role='prefill', capture_seconds=3., capture_prewarmed=True,
                    capture_until_close=True, profile_stop_at_exit=True,
                    nsys_cuda_flush_ms=300000, profile_finalization_timeout=300)
                prof['native_transport_limits'] = dict(
                    prof.get('native_transport_limits', {}), shutdown_timeout=300)
                if model == 'huginn':
                    prof.update(capture_seconds=0., autonomous_capture_seconds=3.,
                        capture_arrival_fraction=.35, capture_prewarmed=False,
                        capture_until_close=False, profile_stop_at_exit=False)
                name = put(prof, count, rate, split='confirmation')
                write(ROOT/'state/joint_confirm.json', dict(stage='SAME_RUN_CAPTURE', name=name, time=time.time()))
                row.update(name=name, complete=execute(name))
                if not row['complete']:
                    row['status'] = 'PROFILE_FAILED'
                    ledger.append(row)
                    break
                with quiet_analysis(name+'_profile'):
                    row['profile'] = analyze(name, clean)
                    row['gates'] = one(prof)
                    row['queue_audit'] = audit(prof, row['gates'])
                checked = verify(name)
                row['captured_probe_correctness'] = checked
                g, q = row['gates'], row['queue_audit'] or {}
                events = ROOT/'raw'/name/'D_probe_requests.jsonl'
                probes = list(map(json.loads, events.read_text().splitlines())) if events.exists() else []
                central = [r for r in probes if g.get('observation_start', float('inf')) <= r['time'] <= r['finish'] <= g.get('observation_end', -1)]
                device_service_screen = bool((g.get('P_actual_kernel_busy_central') or 0) >= .8 and
                    (g.get('P_fraction_reference') or 0) >= .85 and
                    (q.get('half_growth') or 0) > 0 and (q.get('maximum') or 0) >= 2)
                rest = bool(device_service_screen and g.get('D_SLO_gate') and g.get('D_credit_gate') and
                    g.get('D_credit_refusal_sample_fraction') is not None and
                    g['D_credit_refusal_sample_fraction'] <= .5 and central and
                    checked.get('exact') and checked.get('full_work'))
                passed = bool(rest and row['profile'].get('eligible_for_matched_clean'))
                row.update(joint_screen_pass=passed, completed_probes_in_window=len(central),
                    status='JOINT_FINITE_SCREEN_PASS' if passed else 'JOINT_SCREEN_NOT_CONFIRMED',
                    scope='Kernel-union and finite-service screen, not hardware-counter proof or infinite sustainable capacity. Independent seeds remain separate.')
                ledger.append(row)
                write(ROOT/'state/joint_confirm_results.json', ledger)
                # A longer matched cohort is allowed only to reduce the
                # fraction of fixed collector overhead. No SLO gate changes.
                if passed or count == 256 or not rest:
                    break
            write(ROOT/'state/joint_confirm_results.json', ledger)
    write(ROOT/'state/joint_confirm_results.json', ledger)
    write(ROOT/'state/joint_confirm.json', dict(stage='COMPLETE_REVIEW_REQUIRED', time=time.time()))


if __name__ == '__main__':
    try:
        main()
    except BaseException as error:
        write(ROOT/'state/joint_confirm.json', dict(stage='FAILED_REVIEW_REQUIRED', error=repr(error), time=time.time()))
        raise
