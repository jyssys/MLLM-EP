"""Promote exact bulk bookkeeping only after matched native serving controls."""
import copy
from .common import *
from .prepare import put
from .run import execute
from .aligned_gates import one
from .verify_long_probes import verify
from .ready_queue_audit import audit
from .campaign import gate
from .analysis_lane import quiet_analysis


def main():
    terminal = ROOT/'state/bulk_written_controls.json'
    while time.time() < LONG_START_CUTOFF:
        if terminal.exists() and json.loads(terminal.read_text())['stage'] in (
                'COMPLETE_REVIEW_REQUIRED', 'FAILED_REVIEW_REQUIRED'):
            break
        time.sleep(5)
    activate()
    from ..looped_pd_atomic_prefill import evidence, analyze as serving
    from ..looped_pd_headroom_recovery.profile_analysis import analyze
    results = json.loads((ROOT/'state/bulk_written_control_results.json').read_text())
    ledger = []
    for model in ('huginn', 'ouro'):
        controls = [r for r in results if r.get('model') == model]
        if len(controls) != 4 or not all(r.get('passed') for r in controls):
            ledger.append(dict(model=model, status='BLOCKED_CONTROLS'))
            continue
        source = f'bulk_written_{model}_bulk_clean24'
        cfg = json.loads((ROOT/'configs'/(source+'.json')).read_text())
        reference = f'bulk_written_{model}_gate8'
        warm = json.loads((ROOT/'configs'/(reference+'.json')).read_text())
        warm.update(name=f'bulk_written_{model}_probe_warmup_gate8', D_probe_warmup_tokens=16)
        wname = put(warm, 8)
        write(ROOT/'state/bulk_followup.json', dict(stage='PROBE_WARMUP_CORRECTNESS', name=wname, time=time.time()))
        if not execute(wname) or not gate(wname, reference):
            ledger.append(dict(model=model, name=wname, status='BLOCKED_WARMUP_CORRECTNESS'))
            continue
        local = copy.deepcopy(cfg)
        local.update(name=f'capacity_bulk_written_{model}_matched', worker='capacity_worker',
                     capacity_mode='P_ONLY', P_graph_local_capacity=True,
                     jobs=[dict(batch=b, cap=1, repeats=3) for b in (1, 2)])
        lname = put(local, 2, split='confirmation')
        write(ROOT/'state/bulk_followup.json', dict(stage='CAPACITY', name=lname, time=time.time()))
        if time.time() >= LONG_START_CUTOFF or not execute(lname):
            ledger.append(dict(model=model, status='CAPACITY_UNAVAILABLE'))
            continue
        cohorts = [json.loads(line) for line in (ROOT/'raw'/lname/'cohorts.jsonl').read_text().splitlines()]
        rates = [r['job']['batch']/r['prefill_host_service_s'] for r in cohorts if r.get('status') == 'COMPLETE']
        if not rates:
            continue
        # The first pressure level is prescribed from finite measured P service.
        # It is not selected from a future held-out latency or output length.
        rate = round(1.1*max(rates), 3)
        cfg['D_probe_warmup_tokens'] = 16
        confirmed = False
        for start in (1, 2, 3):
            if time.time() >= LONG_START_CUTOFF or (start > 1 and not confirmed):
                break
            c = copy.deepcopy(cfg)
            pairs = ((4, 5), (6, 7), (4, 5)) if model == 'huginn' else ((0, 1), (2, 3), (0, 1))
            c.update(name=f'bulk_joint_{model}_start{start}_clean128', phase='final_repeat',
                     timeout=900, arrival_seed=2026110500+start, independent_start=start,
                     gpus=list(pairs[start-1]), observe_P_submission_readiness=True)
            name = put(c, 128, rate, split='confirmation')
            write(ROOT/'state/bulk_followup.json', dict(stage='CLEAN_128', name=name, time=time.time()))
            if not execute(name) or not evidence.full_work(name)['passed']:
                ledger.append(dict(model=model, name=name, status='BLOCKED_CLEAN'))
                break
            with quiet_analysis(name+'_metrics'):
                metrics, _, _ = serving.serving_one(c)
            if not (metrics['request_TPOT_ms_p95'] <= .9*metrics['TPOT_threshold_ms'] and
                    metrics['token_ITL_ms_p99'] <= metrics['ITL_threshold_ms']):
                ledger.append(dict(model=model, name=name, status='D_SLO_FAILED_NO_PROBE', metrics=metrics))
                break
            p = copy.deepcopy(c)
            p.update(name=f'bulk_joint_{model}_start{start}_probe16_128',
                     D_probe_tokens=16, D_probe_count=8, D_probe_interval_s=2.)
            pname = put(p, 128, rate, split='confirmation')
            write(ROOT/'state/bulk_followup.json', dict(stage='REAL_PROBE', name=pname, time=time.time()))
            if not execute(pname) or not evidence.compare(name, pname)['passed']:
                ledger.append(dict(model=model, name=pname, status='BLOCKED_PROBE'))
                break
            correctness = verify(pname)
            if not correctness.get('exact') or not correctness.get('full_work'):
                ledger.append(dict(model=model, name=pname, status='BLOCKED_REFERENCE', correctness=correctness))
                break
            q = copy.deepcopy(p)
            q.update(name=pname+'_nsys', P_only_profiler=True, nsys=True, capture_role='prefill',
                     capture_seconds=3., capture_prewarmed=True, capture_until_close=True,
                     profile_stop_at_exit=True, nsys_cuda_flush_ms=300000,
                     matched_clean=pname, profile_finalization_timeout=300)
            if model == 'huginn':
                q.update(capture_seconds=0., autonomous_capture_seconds=3.,
                         capture_arrival_fraction=.25, capture_prewarmed=False,
                         capture_until_close=False, profile_stop_at_exit=False)
            q['native_transport_limits'] = dict(q.get('native_transport_limits', {}), shutdown_timeout=300)
            qname = put(q, 128, rate, split='confirmation')
            write(ROOT/'state/bulk_followup.json', dict(stage='SAME_RUN_CAPTURE', name=qname, time=time.time()))
            record = dict(model=model, name=qname, correctness=correctness)
            confirmed = False
            if execute(qname):
                with quiet_analysis(qname+'_profile_and_queue'):
                    record['profile'] = analyze(qname, pname)
                    g = one(q)
                    record['gates'] = g
                    checked = audit(q, g)
                    record['queue_audit'] = checked
                events = ROOT/'raw'/qname/'D_probe_requests.jsonl'
                probes = [json.loads(line) for line in events.read_text().splitlines()] if events.exists() else []
                reference = {r['seed']: r for r in map(json.loads, (ROOT/'raw'/pname/'D_probe_requests.jsonl').read_text().splitlines())}
                exact = bool(probes) and all(r['seed'] in reference and all(
                    r[key] == reference[r['seed']][key] for key in ('input_ids', 'output_ids', 'exit_depths')) for r in probes)
                central = [r for r in probes if g.get('observation_start', float('inf')) <= r['time'] <= r['finish'] <= g.get('observation_end', -1)]
                if probes and not exact:
                    # Timing can admit more probes than the clean run. Check
                    # every captured seed against its own native reference.
                    captured_check = verify(qname)
                    record['captured_probe_reference'] = captured_check
                    exact = bool(captured_check.get('exact') and captured_check.get('full_work'))
                confirmed = bool(checked and checked['audited_P_gate'] and g.get('D_SLO_gate') and
                                 g.get('D_credit_gate') and exact and central)
                record.update(joint_screen_pass=confirmed, completed_probes_in_window=len(central),
                              captured_probe_exact=exact,
                              scope='Finite-window screen. Kernel union is not physical compute occupancy. No sustainable capacity claim.')
            ledger.append(record)
            write(ROOT/'state/bulk_followup_results.json', ledger)
        write(ROOT/'state/bulk_followup_results.json', ledger)
    write(ROOT/'state/bulk_followup.json', dict(stage='COMPLETE_REVIEW_REQUIRED', results=ledger, time=time.time()))


if __name__ == '__main__':
    try:
        main()
    except BaseException as error:
        write(ROOT/'state/bulk_followup.json', dict(stage='FAILED_REVIEW_REQUIRED', error=repr(error), time=time.time()))
        raise
