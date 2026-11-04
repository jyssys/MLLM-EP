"""Test a measured logging bottleneck without changing numerical execution."""
import copy
import time
from .common import *
from .run import execute
from .recovery_qualification import save


def main():
    initialize()
    diagnostic = 'recovery_huginn_graph4_evidence_io_diagnostic512'
    pending = ROOT / 'state' / (diagnostic + '_result.json')
    target = ROOT / 'state/evidence_routing_control.json'
    write(target, dict(status='WAITING_FOR_TARGETED_IO_EVIDENCE', diagnostic=diagnostic))
    while time.time() < LONG_START_CUTOFF and not pending.exists():
        time.sleep(5)
    if not pending.exists() or not json.loads(pending.read_text()).get('complete'):
        write(target, dict(status='DIAGNOSTIC_INCOMPLETE'))
        return
    from .ipc_analysis import analyze
    measured = analyze(diagnostic)
    io = [row for row in measured['groups'] if row['method'].startswith('evidence_')]
    if not any(row['max_seconds'] >= .02 for row in io):
        write(target, dict(status='NO_MEASURED_IO_PAUSE', measured=measured))
        return
    from ..looped_pd_autonomous_headroom.analysis_lane import quiet_analysis
    from ..looped_pd_autonomous_headroom.campaign import gate
    from ..looped_pd_atomic_prefill import evidence
    from .state_comparison import compare
    from .geometry_confirmation import pair
    reference = 'recovery_huginn_graph_state8'
    base_state = json.loads((ROOT / 'configs' / (reference + '.json')).read_text())
    base = json.loads((ROOT / 'configs/recovery_huginn_slots4_graph4_Ponly_joint_s1_clean512.json').read_text())
    results = []
    for slots, rows in ((4, 4), (8, 8)):
        label = f'async_evidence_slots{slots}_graph{rows}'
        cfg = copy.deepcopy(base_state)
        cfg.update(name=f'recovery_huginn_{label}_state8', D_slots=slots, graph_rows=rows,
                   route_campaign_evidence_async=True, observe_native_graph_coverage=True)
        save(cfg)
        record = dict(label=label, state_run=cfg['name'], trials=[])
        complete = execute(cfg['name'])
        if complete:
            with quiet_analysis(cfg['name'] + '_state_audit'):
                record.update(state_gate=gate(cfg['name']),
                              state_parity=compare(reference, cfg['name']),
                              output=evidence.compare(reference, cfg['name']))
        if not complete or not (record['state_gate'] and record['state_parity']['exact'] and record['output']['passed']):
            record['status'] = 'BLOCKED_CORRECTNESS'
            results.append(record)
            write(target, dict(status='IN_PROGRESS', io_evidence=io, results=results))
            continue
        qualified = copy.deepcopy(base)
        qualified.update(D_slots=slots, graph_rows=rows, route_campaign_evidence_async=True)
        for key in ('observe_host_pauses', 'observe_evidence_io', 'observe_python_gc'):
            qualified.pop(key, None)
        reference_name = 'capacity_recovery_huginn_async_evidence_L2048'
        reference_result = ROOT / 'raw' / reference_name / 'dispatcher_result.json'
        if not reference_result.exists():
            capacity = json.loads((ROOT / 'configs/capacity_recovery_huginn_L2048.json').read_text())
            capacity.update(name=reference_name, route_campaign_evidence_async=True)
            save(capacity)
            if not execute(reference_name):
                record['status'] = 'REFERENCE_EXECUTION_FAILED'
                results.append(record)
                break
        elif not json.loads(reference_result.read_text()).get('complete'):
            record['status'] = 'REFERENCE_EXECUTION_FAILED'
            results.append(record)
            break
        for start in (1, 2, 3):
            if time.time() >= LONG_START_CUTOFF:
                break
            trial = pair(qualified, label, start)
            record['trials'].append(trial)
            write(target, dict(status='IN_PROGRESS', io_evidence=io, results=results, current=record))
            if not trial['passed']:
                break
        results.append(record)
        passed = len(record['trials']) == 3 and all(row['passed'] for row in record['trials'])
        if passed:
            write(ROOT / 'state/huginn_final_recovery_gate.json', dict(
                status='RECOVERY_READY', same_run_gates=True, whole_run_decode_SLO=True,
                independent_repeats=True, exact_state_and_output=True,
                qualified_configuration=qualified, geometry_state_gate=record,
                results=record['trials'],
                scope='Operational kernel-union and finite service gates. No peak-FLOP or useful migrated-work claim.'))
            write(target, dict(status='RECOVERY_READY', io_evidence=io, results=results))
            return
    write(target, dict(status='RECOVERY_FAILED_OR_INCOMPLETE', io_evidence=io, results=results))


if __name__ == '__main__':
    main()
