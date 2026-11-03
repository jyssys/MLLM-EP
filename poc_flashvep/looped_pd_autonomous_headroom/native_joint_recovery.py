"""Test retained native GPU events on warmed, exact 128-request probe runs."""
import copy
from .common import *
from .prepare import put
from .run import execute
from .verify_long_probes import verify
from .aligned_gates import one
from .ready_queue_audit import audit
from .analysis_lane import quiet_analysis


def main():
    activate()
    from ..looped_pd_atomic_prefill import evidence, analyze as serving
    from ..looped_pd_headroom_recovery.cuda_only_profile import analyze
    smoke = json.loads((ROOT/'state/native_start_stop_control.json').read_text())['result']
    assert smoke.get('complete') and smoke.get('profile', {}).get('status') == 'PARSED'
    ledger = []
    for model in ('huginn', 'ouro'):
        source = f'bulk_joint_{model}_start1_probe16_128'
        while time.time() < LONG_START_CUTOFF:
            status = ROOT/'raw'/source/'dispatcher_result.json'
            if status.exists():
                break
            parent = ROOT/'state/bulk_followup.json'
            if parent.exists() and json.loads(parent.read_text())['stage'] in ('COMPLETE_REVIEW_REQUIRED', 'FAILED_REVIEW_REQUIRED'):
                break
            time.sleep(5)
        if not status.exists() or not json.loads(status.read_text()).get('complete'):
            ledger.append(dict(model=model, status='NO_COMPLETED_WARMED_PROBE_SOURCE'))
            continue
        original = json.loads((ROOT/'configs'/(source+'.json')).read_text())
        confirmed = False
        for start in (1, 2, 3):
            if time.time() >= LONG_START_CUTOFF or (start > 1 and not confirmed):
                break
            cfg = copy.deepcopy(original)
            clean = source
            if start > 1:
                cfg.update(name=f'native_joint_{model}_start{start}_baseline128',
                    arrival_seed=2026110600+start, independent_start=start,
                    gpus=list((6, 7) if model == 'huginn' and start == 2 else
                              (4, 5) if model == 'huginn' else (2, 3) if start == 2 else (0, 1)))
                for key in ('D_probe_tokens', 'D_probe_count', 'D_probe_interval_s'):
                    cfg.pop(key, None)
                baseline = put(cfg, 128, cfg['arrival_rate'], split='confirmation')
                write(ROOT/'state/native_joint_recovery.json', dict(stage='CLEAN_REPEAT', name=baseline, time=time.time()))
                if not execute(baseline) or not evidence.full_work(baseline)['passed']:
                    break
                with quiet_analysis(baseline+'_SLO'):
                    metrics, _, _ = serving.serving_one(cfg)
                if metrics['request_TPOT_ms_p95'] > .9*metrics['TPOT_threshold_ms'] or metrics['token_ITL_ms_p99'] > metrics['ITL_threshold_ms']:
                    ledger.append(dict(model=model, name=baseline, status='D_BASELINE_SLO_FAILED', metrics=metrics))
                    break
                cfg.update(name=f'native_joint_{model}_start{start}_probe16_128',
                           D_probe_tokens=16, D_probe_count=8, D_probe_interval_s=2.)
                clean = put(cfg, 128, cfg['arrival_rate'], split='confirmation')
                write(ROOT/'state/native_joint_recovery.json', dict(stage='PROBE_REPEAT', name=clean, time=time.time()))
                if not execute(clean) or not evidence.compare(baseline, clean)['passed']:
                    break
                check = verify(clean)
                if not check.get('exact') or not check.get('full_work'):
                    ledger.append(dict(model=model, name=clean, status='BLOCKED_PROBE_CORRECTNESS', correctness=check))
                    break
            prof = copy.deepcopy(cfg)
            prof.update(name=f'native_joint_{model}_start{start}_bounded128',
                phase='bounded_native_joint_capture', matched_clean=clean,
                bounded_native_capture_seconds=6., capture_arrival_fraction=.35,
                cuda_only_profile=False, P_only_profiler=False, nsys=False,
                capture_seconds=0., autonomous_capture_seconds=0., postwarm_capture=False,
                capture_prewarmed=False, capture_until_close=False, capture_role='prefill',
                profile_finalization_timeout=300)
            if (ROOT/'raw'/prof['name']/'supervisor_recovery.json').exists():
                prof['name'] += '_supervised_retry'
            name = put(prof, 128, cfg['arrival_rate'], split='confirmation')
            write(ROOT/'state/native_joint_recovery.json', dict(stage='SAME_RUN_NATIVE_CAPTURE', name=name, time=time.time()))
            row = dict(model=model, name=name, complete=execute(name), start=start)
            confirmed = False
            if row['complete']:
                with quiet_analysis(name+'_profile_and_queue'):
                    row['profile'] = analyze(name, clean)
                    if row['profile'].get('status') == 'PARSED':
                        row['gates'] = one(prof)
                        row['queue_audit'] = audit(prof, row['gates'])
                check = verify(name)
                row['correctness'] = check
                g, checked = row.get('gates', {}), row.get('queue_audit')
                path = ROOT/'raw'/name/'D_probe_requests.jsonl'
                probes = list(map(json.loads, path.read_text().splitlines())) if path.exists() else []
                central = [p for p in probes if g.get('observation_start', float('inf')) <= p['time'] <= p['finish'] <= g.get('observation_end', -1)]
                confirmed = bool(checked and checked['audited_P_gate'] and g.get('D_SLO_gate') and
                    g.get('D_credit_gate') and central and check.get('exact') and check.get('full_work'))
                row.update(joint_screen_pass=confirmed, completed_probes_in_window=len(central),
                    scope='Six-second GPU window within a 128-request run. Not continuous whole-run GPU evidence or sustainable capacity.')
            ledger.append(row)
            write(ROOT/'state/native_joint_recovery_results.json', ledger)
        write(ROOT/'state/native_joint_recovery_results.json', ledger)
    write(ROOT/'state/native_joint_recovery.json', dict(stage='COMPLETE_REVIEW_REQUIRED', time=time.time()))


if __name__ == '__main__':
    try:
        main()
    except BaseException as error:
        write(ROOT/'state/native_joint_recovery.json', dict(stage='FAILED_REVIEW_REQUIRED', error=repr(error), time=time.time()))
        raise
