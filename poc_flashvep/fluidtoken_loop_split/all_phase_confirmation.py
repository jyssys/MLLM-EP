"""Test probe admission across all Decode phases after the P-only control."""
import copy
import time
from .common import *
from .geometry_confirmation import pair


def main():
    initialize()
    target = ROOT / 'state/all_phase_confirmation.json'
    write(ROOT / 'state/all_phase_confirmation_plan.json', dict(
        hypothesis='The recurrent-only guard can admit a probe while other requests occupy Prelude or Coda.',
        intervention='Count every active Decode phase. Keep full probe work and every acceptance threshold.',
        order=['slots4 at 1.971', 'slots8 at 1.971', 'slots4 at 1.85'],
        lower_rate_reason='If credit remains limiting, test a lower offered rate that still exceeds the measured P reference.',
        exclusion='Do not add probe outputs to serving throughput. Do not accept zero probes or combine runs.'))
    pending = ROOT / 'state/evidence_p_only_confirmation.json'
    write(target, dict(status='WAITING_FOR_P_ONLY_CONTROL'))
    while time.time() < LONG_START_CUTOFF:
        if pending.exists():
            state = json.loads(pending.read_text())
            if state['status'] == 'RECOVERY_READY':
                write(target, dict(status='NOT_NEEDED'))
                return
            if state['status'] == 'RECOVERY_FAILED_OR_INCOMPLETE':
                break
        time.sleep(5)
    else:
        write(target, dict(status='NOT_RUN_TIME_BUDGET'))
        return
    previous = json.loads((ROOT / 'state/evidence_routing_control.json').read_text())
    candidates = {row['label']: row for row in previous['results']}
    results = []
    for slots, rate in ((4, 1.971), (8, 1.971), (4, 1.85)):
        candidate = candidates[f'async_evidence_slots{slots}_graph{slots}']
        if not (candidate.get('state_gate') and candidate.get('state_parity', {}).get('exact')):
            continue
        path = ROOT / 'configs' / f'recovery_huginn_async_evidence_slots{slots}_graph{slots}_joint_s1_clean512.json'
        base = copy.deepcopy(json.loads(path.read_text()))
        base.update(D_probe_admission_scope='all_decode_phases', arrival_rate=rate)
        label = f'allphase_slots{slots}_q{str(rate).replace(".", "p")}'
        trials = []
        for start in (1, 2, 3):
            if time.time() >= LONG_START_CUTOFF:
                break
            trial = pair(base, label, start, roles=('prefill',))
            trials.append(trial)
            write(target, dict(status='IN_PROGRESS', results=results, current=trials))
            if not trial['passed']:
                break
        results.append(dict(label=label, trials=trials))
        if len(trials) == 3 and all(row['passed'] for row in trials):
            write(ROOT / 'state/huginn_final_recovery_gate.json', dict(
                status='RECOVERY_READY', same_run_gates=True, whole_run_decode_SLO=True,
                independent_repeats=True, exact_state_and_output=True,
                qualified_configuration=base, geometry_state_gate=candidate, results=trials,
                D_kernel_activity='NOT_MEASURED_IN_ACCEPTED_RUNS',
                scope='P-only kernel capture. Exact D probes, credit, and Decode SLO are measured in the same execution.'))
            write(target, dict(status='RECOVERY_READY', results=results))
            from .conditional_split_qualification import main as qualify
            from .static_screen import main as screen
            qualify()
            screen()
            return
    write(target, dict(status='RECOVERY_FAILED_OR_INCOMPLETE', results=results))


if __name__ == '__main__':
    main()
