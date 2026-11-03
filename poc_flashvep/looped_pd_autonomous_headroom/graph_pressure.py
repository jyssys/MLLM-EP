"""Choose pressure from measured native-graph P service, not arbitrary QPS."""
import copy
from .common import *
from .prepare import put
from .run import execute
from .aligned_gates import one


def main():
    activate()
    from ..looped_pd_atomic_prefill import evidence
    from ..looped_pd_headroom_recovery.profile_analysis import analyze
    ledger = []
    for model in ('ouro', 'huginn'):
        source = f'prefill_graph_{model}_graph_independent32'
        while time.time() < LONG_START_CUTOFF:
            state = ROOT/'state/prefill_graph_followup_results.json'
            records = json.loads(state.read_text()) if state.exists() else []
            entry = next((r for r in records if r['name'] == source), None)
            capacity = ROOT/'raw'/f'capacity_prefill_graph_{model}_matched'
            if entry is not None and not entry['passed']:
                break
            if entry is not None and (capacity/'dispatcher_result.json').exists():
                break
            terminal = ROOT/'state/prefill_graph_followup.json'
            if terminal.exists() and json.loads(terminal.read_text())['stage'] in ('COMPLETE_REVIEW_REQUIRED', 'FAILED_REVIEW_REQUIRED'):
                break
            time.sleep(5)
        if entry is None or not entry['passed'] or not (capacity/'cohorts.jsonl').exists():
            ledger.append(dict(model=model, status='BLOCKED_PREDECESSOR'))
            continue
        cohorts = [json.loads(line) for line in (capacity/'cohorts.jsonl').read_text().splitlines()]
        rates = [r['job']['batch']/r['prefill_host_service_s'] for r in cohorts if r.get('status') == 'COMPLETE']
        if not rates:
            continue
        reference = max(rates)
        for multiplier in (1.1, 1.3):
            cfg = copy.deepcopy(json.loads((ROOT/'configs'/(source+'.json')).read_text()))
            rate = round(reference*multiplier, 3)
            tag = f'graph_pressure_{model}_factor{int(multiplier*100)}'
            cfg.update(name=tag, phase='qps_pilot', timeout=600)
            name = put(cfg, 24, rate, split='confirmation')
            write(ROOT/'state/graph_pressure.json', dict(stage='CLEAN', name=name, reference=reference,
                  reference_scope='Measured finite P service. Not sustainable capacity.', multiplier=multiplier, time=time.time()))
            if not execute(name):
                ledger.append(dict(name=name, status='EXECUTION_FAILED'))
                break
            requests = [json.loads(line) for line in (ROOT/'raw'/name/'requests.jsonl').read_text().splitlines()]
            prior = [json.loads(line) for line in (ROOT/'raw'/source/'requests.jsonl').read_text().splitlines()]
            byid = {r['sample_id']: r for r in prior if not r.get('warmup')}
            exact = all(byid[r['sample_id']]['input_ids'] == r['input_ids'] and
                        byid[r['sample_id']]['seed'] == r['seed'] and
                        byid[r['sample_id']]['output_ids'] == r['output_ids']
                        for r in requests if not r.get('warmup'))
            if not exact or not evidence.full_work(name)['passed']:
                ledger.append(dict(name=name, status='BLOCKED_CORRECTNESS'))
                break
            prof = copy.deepcopy(cfg)
            prof.update(name=name+('_nsys' if model == 'ouro' else '_nsys_bounded_retry'), phase='bounded_stage_profiling', P_only_profiler=True,
                        nsys=True, capture_role='prefill', capture_seconds=3., capture_prewarmed=model == 'ouro',
                        capture_until_close=model == 'ouro', profile_stop_at_exit=model == 'ouro', nsys_cuda_flush_ms=300000,
                        matched_clean=name, profile_finalization_timeout=300)
            prof['native_transport_limits'] = dict(prof.get('native_transport_limits', {}), shutdown_timeout=300)
            pname = put(prof, 24, rate, split='confirmation')
            write(ROOT/'state/graph_pressure.json', dict(stage='PROFILE', name=pname, time=time.time()))
            record = dict(model=model, name=name, rate=rate, finite_P_reference=reference,
                          multiplier=multiplier, matched_outputs=exact)
            if execute(pname):
                record['profile'] = analyze(pname, name)
                record['gates'] = one(prof)
            ledger.append(record)
            write(ROOT/'state/graph_pressure_results.json', ledger)
            if record.get('gates', {}).get('P_compute_gate') and record['gates'].get('D_SLO_gate'):
                break
    write(ROOT/'state/graph_pressure.json', dict(stage='COMPLETE_REVIEW_REQUIRED', results=ledger, time=time.time()))


if __name__ == '__main__':
    try:
        main()
    except BaseException as error:
        write(ROOT/'state/graph_pressure.json', dict(stage='FAILED_REVIEW_REQUIRED', error=repr(error), time=time.time()))
        raise
