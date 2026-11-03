"""Start native collection before graph warm-up, without dynamic attachment."""
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
    from ..looped_pd_atomic_prefill import evidence
    from ..looped_pd_headroom_recovery.cuda_only_profile import analyze
    ledger = []
    for model in ('huginn', 'ouro'):
        path = ROOT/'state/d_graph_state_followup_results.json'
        while time.time() < LONG_START_CUTOFF:
            rows = json.loads(path.read_text()) if path.exists() else []
            selected = [r for r in rows if r.get('model') == model]
            if selected:
                break
            state = ROOT/'state/d_graph_state_followup.json'
            if state.exists() and json.loads(state.read_text())['stage'] in ('COMPLETE_REVIEW_REQUIRED', 'FAILED_REVIEW_REQUIRED'):
                break
            time.sleep(5)
        if not selected or not selected[0].get('probe_correctness', {}).get('exact'):
            ledger.append(dict(model=model, status='NO_VERIFIED_D_GRAPH_SOURCE'))
            continue
        source = selected[0]['name']
        original = json.loads((ROOT/'configs'/(source+'.json')).read_text())
        for count in (24, 128):
            if time.time() >= LONG_START_CUTOFF:
                break
            cfg = copy.deepcopy(original)
            if count == 24:
                cfg['name'] = f'early_native_{model}_matched_clean24'
                clean = put(cfg, count, cfg['arrival_rate'], split='confirmation')
                if not execute(clean):
                    break
            else:
                clean = source
            cfg.update(name=f'early_native_{model}_profile{count}',
                phase='collector_attach_timing_control', matched_clean=clean,
                cuda_only_profile=True, capture_role='prefill', capture_roles=['prefill'],
                native_capture_seconds=0., bounded_native_capture_seconds=0.,
                native_capture_stop_at_close=False, P_only_profiler=False, nsys=False,
                capture_seconds=0., autonomous_capture_seconds=0., postwarm_capture=False,
                capture_prewarmed=False, capture_until_close=False, profile_finalization_timeout=300)
            name = put(cfg, count, cfg['arrival_rate'], split='confirmation')
            write(ROOT/'state/native_early_capture_control.json', dict(stage='RUNNING', name=name, time=time.time()))
            row = dict(name=name, model=model, count=count, complete=execute(name),
                hypothesis='Collection before graph capture and transport warm-up avoids attaching to active graph/transfer work.')
            if row['complete']:
                with quiet_analysis(name+'_profile'):
                    row['profile'] = analyze(name, clean)
                    if row['profile'].get('status') == 'PARSED':
                        row['gates'] = one(cfg)
                        row['queue_audit'] = audit(cfg, row['gates'])
                row['probe_correctness'] = verify(name)
                g, q = row.get('gates', {}), row.get('queue_audit') or {}
                check = row['probe_correctness']
                probes = [r for r in map(json.loads, (ROOT/'raw'/name/'D_probe_requests.jsonl').read_text().splitlines())]
                central = [r for r in probes if g.get('observation_start', float('inf')) <= r['time'] <= r['finish'] <= g.get('observation_end', -1)]
                row.update(completed_probes_in_window=len(central),
                    joint_screen_pass=bool(count >= 128 and q.get('audited_P_gate') and g.get('D_SLO_gate')
                        and g.get('D_credit_gate') and central and check.get('exact') and check.get('full_work')))
            ledger.append(row)
            write(ROOT/'state/native_early_capture_results.json', ledger)
            if not row['complete'] or not row.get('profile', {}).get('exact_output_parity'):
                break
    write(ROOT/'state/native_early_capture_results.json', ledger)
    write(ROOT/'state/native_early_capture_control.json', dict(stage='COMPLETE_REVIEW_REQUIRED', time=time.time()))


if __name__ == '__main__':
    try:
        main()
    except BaseException as error:
        write(ROOT/'state/native_early_capture_control.json', dict(stage='FAILED_REVIEW_REQUIRED', error=repr(error), time=time.time()))
        raise
