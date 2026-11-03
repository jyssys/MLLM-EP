"""Test P pressure and real D work in one run after native graph qualification."""
import copy
from .common import *
from .prepare import put
from .run import execute
from .aligned_gates import one
from .verify_long_probes import verify


def main():
    while time.time() < LONG_START_CUTOFF:
        path = ROOT/'state/graph_pressure.json'
        if path.exists() and json.loads(path.read_text())['stage'] in ('COMPLETE_REVIEW_REQUIRED', 'FAILED_REVIEW_REQUIRED'):
            break
        time.sleep(5)
    path = ROOT/'state/graph_pressure_results.json'
    records = json.loads(path.read_text()) if path.exists() else []
    activate()
    from ..looped_pd_atomic_prefill import evidence, analyze as serving_analysis
    from ..looped_pd_headroom_recovery.profile_analysis import analyze
    ledger = []
    for model in ('ouro', 'huginn'):
        candidates = [r for r in records if r.get('model') == model and r.get('matched_outputs') and
                      r.get('gates', {}).get('D_credit_gate')]
        if not candidates:
            ledger.append(dict(model=model, status='NO_QUALIFIED_PILOT_CANDIDATE'))
            continue
        selected = max(candidates, key=lambda r: (r['gates'].get('P_compute_gate', False),
                         r['gates'].get('D_SLO_gate', False),
                         r['profile'].get('eligible_for_matched_clean', False),
                         r['gates'].get('P_actual_kernel_busy_central') or 0))
        original = json.loads((ROOT/'configs'/(selected['name']+'.json')).read_text())
        confirmed = False
        for start in (1, 2, 3):
            if time.time() >= LONG_START_CUTOFF or (start > 1 and not confirmed):
                break
            cfg = copy.deepcopy(original)
            pairs = ((0, 1), (2, 3), (0, 1)) if model == 'ouro' else ((4, 5), (6, 7), (4, 5))
            cfg.update(name=f'graph_joint_{model}_start{start}_clean128', phase='final_repeat',
                       arrival_seed=2026110400+start, independent_start=start, timeout=900,
                       gpus=list(pairs[start-1]), GPU_pair_crossover=True,
                       observe_P_submission_readiness=True)
            name = put(cfg, 128, cfg['arrival_rate'], split='confirmation')
            write(ROOT/'state/graph_joint_validation.json', dict(stage='CLEAN_128', name=name, time=time.time()))
            if not execute(name) or not evidence.full_work(name)['passed']:
                ledger.append(dict(name=name, status='BLOCKED_CLEAN_OR_WORK'))
                break
            baseline_metrics, _, _ = serving_analysis.serving_one(cfg)
            baseline_safe = (baseline_metrics['request_TPOT_ms_p95'] <= .9*baseline_metrics['TPOT_threshold_ms']
                             and baseline_metrics['token_ITL_ms_p99'] <= baseline_metrics['ITL_threshold_ms'])
            if not baseline_safe:
                ledger.append(dict(name=name, status='D_BASELINE_SLO_FAILED_NO_PROBE', metrics=baseline_metrics))
                write(ROOT/'state/graph_joint_validation_results.json', ledger)
                break
            probe = copy.deepcopy(cfg)
            probe.update(name=f'graph_joint_{model}_start{start}_probe16_128',
                         D_probe_tokens=16, D_probe_count=8, D_probe_interval_s=2.)
            pname = put(probe, 128, cfg['arrival_rate'], split='confirmation')
            write(ROOT/'state/graph_joint_validation.json', dict(stage='REAL_PROBE_128', name=pname, time=time.time()))
            if not execute(pname) or not evidence.compare(name, pname)['passed']:
                ledger.append(dict(name=pname, status='BLOCKED_PROBE_OUTPUT'))
                break
            correctness = verify(pname)
            if not correctness.get('exact') or not correctness.get('full_work'):
                ledger.append(dict(name=pname, status='BLOCKED_PROBE_REFERENCE', correctness=correctness))
                break
            prof = copy.deepcopy(probe)
            prof.update(name=pname+'_nsys', phase='bounded_stage_profiling', P_only_profiler=True,
                        nsys=True, capture_role='prefill', capture_seconds=3., capture_prewarmed=True,
                        capture_until_close=True, profile_stop_at_exit=True, nsys_cuda_flush_ms=300000,
                        matched_clean=pname, profile_finalization_timeout=300)
            prof['native_transport_limits'] = dict(prof.get('native_transport_limits', {}), shutdown_timeout=300)
            if model == 'huginn':
                # Full-interval collection stalled in two earlier native NIXL
                # runs. Preserve those failures and use a new bounded window.
                prof.update(name=pname+'_nsys_timed', capture_seconds=0.,
                            autonomous_capture_seconds=3., capture_arrival_fraction=.35,
                            capture_prewarmed=False, capture_until_close=False,
                            profile_stop_at_exit=False)
            qname = put(prof, 128, cfg['arrival_rate'], split='confirmation')
            write(ROOT/'state/graph_joint_validation.json', dict(stage='SAME_RUN_CAPTURE_AND_PROBE', name=qname, time=time.time()))
            record = dict(model=model, name=qname, start=start, correctness=correctness)
            confirmed = False
            if execute(qname):
                record['profile'] = analyze(qname, pname)
                g = one(prof)
                record['gates'] = g
                from .ready_queue_audit import audit
                queue_audit = audit(prof, g)
                record['submission_queue_audit'] = queue_audit
                path = ROOT/'raw'/qname/'D_probe_requests.jsonl'
                probes = [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []
                central = [r for r in probes if g.get('observation_start', float('inf')) <= r['time'] <= r['finish'] <= g.get('observation_end', -1)]
                reference_rows = [json.loads(line) for line in (ROOT/'raw'/pname/'D_probe_requests.jsonl').read_text().splitlines()]
                expected = {r['seed']: r for r in reference_rows}
                exact = bool(probes) and all(r['seed'] in expected and r['input_ids'] == expected[r['seed']]['input_ids'] and
                                             r['output_ids'] == expected[r['seed']]['output_ids'] and
                                             r['exit_depths'] == expected[r['seed']]['exit_depths'] for r in probes)
                p_gate = queue_audit['audited_P_gate'] if queue_audit else g.get('P_compute_gate')
                confirmed = bool(p_gate and g.get('D_SLO_gate') and g.get('D_credit_gate') and central and exact)
                record.update(probes_completed_in_same_window=len(central), captured_probe_exact=exact,
                              joint_screen_pass=confirmed,
                              scope='A single finite start is not sustained-capacity proof. Independent repeats and physical-resource evidence remain separate.')
            ledger.append(record)
            write(ROOT/'state/graph_joint_validation_results.json', ledger)
    write(ROOT/'state/graph_joint_validation.json', dict(stage='COMPLETE_REVIEW_REQUIRED', results=ledger, time=time.time()))


if __name__ == '__main__':
    try:
        main()
    except BaseException as error:
        write(ROOT/'state/graph_joint_validation.json', dict(stage='FAILED_REVIEW_REQUIRED', error=repr(error), time=time.time()))
        raise
