"""Independent prompt qualification and matched clean native-graph controls."""
import copy
from .common import *
from .prepare import put
from .run import execute


def main():
    while time.time() < LONG_START_CUTOFF:
        state = ROOT/'state/prefill_graph_controls.json'
        if state.exists() and json.loads(state.read_text())['stage'] in (
                'CONTROLS_COMPLETE_REVIEW_REQUIRED', 'FAILED_REVIEW_REQUIRED'):
            break
        time.sleep(5)
    activate()
    from ..looped_pd_atomic_prefill import evidence
    from ..looped_pd_headroom_recovery.profile_analysis import analyze
    from ..looped_pd_headroom_recovery.gates import one
    controls = json.loads((ROOT/'state/prefill_graph_control_results.json').read_text())
    ledger = []
    for model, rate in [('ouro', 3.), ('huginn', 1.2)]:
        matched = [r for r in controls if r['model'] == model]
        if len(matched) != 2 or not all(r['passed'] for r in matched):
            continue
        reference = None
        for source in matched:
            cfg = copy.deepcopy(json.loads((ROOT/'configs'/(source['name']+'.json')).read_text()))
            label = 'graph' if cfg.get('P_prefill_graph_rows') else 'eager'
            cfg.update(name=f'prefill_graph_{model}_{label}_independent32',
                       verify_transfer=False, state_probe=False, cap=32,
                       phase='independent_correctness_and_clean_control',
                       arrival_seed=2026110351, seed=2026110352)
            name = put(cfg, 32, rate, split='confirmation')
            write(ROOT/'state/prefill_graph_followup.json', dict(stage='RUNNING', name=name, time=time.time()))
            if not execute(name):
                ledger.append(dict(name=name, passed=False, reason='EXECUTION_FAILED'))
                break
            work = evidence.full_work(name)
            parity = evidence.compare(reference, name) if reference else None
            passed = work['passed'] and (parity is None or parity['passed'])
            row = dict(name=name, model=model, full_work=work, comparison=parity, passed=passed)
            ledger.append(row)
            write(ROOT/'state/prefill_graph_followup_results.json', ledger)
            if not passed:
                break
            reference = name
            if label != 'graph':
                continue
            local = copy.deepcopy(cfg)
            local.update(name=f'capacity_prefill_graph_{model}_matched',
                         worker='capacity_worker', capacity_mode='P_ONLY',
                         P_graph_local_capacity=True, phase='independent_capacity',
                         jobs=[dict(batch=b, cap=1, repeats=3) for b in (1, 2)])
            lname = put(local, 2, split='confirmation')
            write(ROOT/'state/prefill_graph_followup.json', dict(stage='MATCHED_CAPACITY', name=lname, time=time.time()))
            if not execute(lname):
                ledger.append(dict(name=lname, passed=False, reason='CAPACITY_FAILED'))
                write(ROOT/'state/prefill_graph_followup_results.json', ledger)
                break
            prof = copy.deepcopy(cfg)
            prof.update(name=name+'_nsys', phase='bounded_stage_profiling',
                        P_only_profiler=True, nsys=True, capture_role='prefill',
                        capture_seconds=3., capture_prewarmed=True,
                        capture_until_close=True, profile_stop_at_exit=True,
                        nsys_cuda_flush_ms=300000, matched_clean=name,
                        profile_finalization_timeout=300)
            prof['native_transport_limits'] = dict(prof.get('native_transport_limits', {}), shutdown_timeout=300)
            pname = put(prof, 32, rate, split='confirmation')
            write(ROOT/'state/prefill_graph_followup.json', dict(stage='PROFILE', name=pname, time=time.time()))
            if execute(pname):
                ledger.append(dict(name=pname, profile=analyze(pname, name), gates=one(prof)))
                write(ROOT/'state/prefill_graph_followup_results.json', ledger)
    write(ROOT/'state/prefill_graph_followup.json',
          dict(stage='COMPLETE_REVIEW_REQUIRED', results=ledger, time=time.time()))


if __name__ == '__main__':
    try:
        main()
    except BaseException as error:
        write(ROOT/'state/prefill_graph_followup.json',
              dict(stage='FAILED_REVIEW_REQUIRED', error=repr(error), time=time.time()))
        raise
