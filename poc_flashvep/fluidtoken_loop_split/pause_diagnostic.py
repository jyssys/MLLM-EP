"""Reproduce residual clean tails with bounded, targeted host observers."""
import copy
import argparse
from .common import *
from .run import execute
from .recovery_qualification import save


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', default='recovery_huginn_graph4_ipc_pause_diagnostic512')
    parser.add_argument('--evidence-io', action='store_true')
    args = parser.parse_args()
    initialize()
    cfg = copy.deepcopy(json.loads((ROOT / 'configs/recovery_huginn_slots4_graph4_Ponly_joint_s1_clean512.json').read_text()))
    cfg.update(name=args.name,
        observe_python_gc=True, observe_host_pauses=True,
        observe_evidence_io=args.evidence_io,
        phase='HOST_PAUSE_DIAGNOSTIC_NOT_CLEAN_PERFORMANCE')
    save(cfg)
    write(ROOT / 'state' / (cfg['name'] + '_plan.json'), dict(
        symptom='The retained clean run has 85 ITL-limit violations. Only nine overlap a Prefill probe. Several requests pause together for about one second.',
        hypotheses=[
            'Python collection or host runtime pauses interrupt submissions or output delivery.',
            'Memory queries or NIXL progress hold a driver or host thread.',
            'Long model submissions expose a remaining warmup or execution-geometry path.',
            'An unobserved system scheduling pause interrupts a worker.'],
        unchanged='Same prompts, arrival trace, slots, graph rows, probes, precision, model, and R32.',
        instrumentation='GC start/stop callbacks and host calls longer than 20 ms. No policy change or GPU synchronization is added.',
        interpretation='Temporal overlap narrows hypotheses but does not prove causation. This is not clean latency evidence.'))
    complete = execute(cfg['name'])
    result = dict(name=cfg['name'], complete=complete)
    if complete:
        from ..looped_pd_autonomous_headroom.analysis_lane import quiet_analysis
        from ..looped_pd_atomic_prefill import evidence
        from .whole_tail_audit import one
        from .probe_reference import verify
        with quiet_analysis(cfg['name'] + '_pause_audit'):
            result['tails'] = one(cfg['name'])
            result['output'] = evidence.compare('recovery_huginn_qualified_admission_no_probe512', cfg['name'])
            result['probes'] = verify(cfg['name'])
            folder = ROOT / 'raw' / cfg['name']
            window = json.loads((folder / 'window.json').read_text())
            events = []
            for pattern, kind in (('python_gc_*.json', 'GC'), ('host_pauses_*.json', 'HOST_CALL')):
                for path in folder.glob(pattern):
                    value = json.loads(path.read_text())
                    events += [dict(event, pid=value['pid'], kind=kind) for event in value['events']
                               if window['start'] <= event['start'] <= window['end']]
            result['largest_host_intervals'] = sorted(events, key=lambda e: e['end'] - e['start'], reverse=True)[:100]
            result['top_ITL_overlap'] = [dict(interval=interval, overlapping_host_intervals=[e for e in events
                if e['start'] < interval['end'] and e['end'] > interval['start']])
                for interval in result['tails']['top_intervals']]
    write(ROOT / 'state' / (cfg['name'] + '_result.json'), result)


if __name__ == '__main__':
    main()
