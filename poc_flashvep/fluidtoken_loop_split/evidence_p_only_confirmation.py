"""Repeat the original P-only activity contract after the evidence-I/O fix."""
import time
from .common import *
from .geometry_confirmation import pair


def main():
    initialize()
    pending = ROOT / 'state/evidence_routing_control.json'
    target = ROOT / 'state/evidence_p_only_confirmation.json'
    write(target, dict(status='WAITING_FOR_DUAL_CONTROL',
        contract='All original Phase A3 gates remain. Also require whole-run Decode SLO and three independent starts.',
        limitation='P-only captures do not measure D kernel union. Use same-run exact probes, KV credit, and Decode SLO. Never import a failed dual trace.'))
    while time.time() < LONG_START_CUTOFF:
        if pending.exists():
            previous = json.loads(pending.read_text())
            if previous['status'] == 'RECOVERY_READY':
                write(target, dict(status='NOT_NEEDED_DUAL_GATE_PASSED'))
                return
            if previous['status'] == 'RECOVERY_FAILED_OR_INCOMPLETE':
                break
            if previous['status'] in ('NO_MEASURED_IO_PAUSE', 'DIAGNOSTIC_INCOMPLETE'):
                write(target, dict(status='BLOCKED_DIAGNOSTIC'))
                return
        time.sleep(5)
    else:
        write(target, dict(status='NOT_RUN_TIME_BUDGET'))
        return
    results = []
    for candidate in previous['results']:
        if not (candidate.get('state_gate') and candidate.get('state_parity', {}).get('exact')
                and candidate.get('output', {}).get('passed')):
            continue
        label = candidate['label']
        path = ROOT / 'configs' / f'recovery_huginn_{label}_joint_s1_clean512.json'
        if not path.exists():
            continue
        base = json.loads(path.read_text())
        trials = []
        for start in (1, 2, 3):
            if time.time() >= LONG_START_CUTOFF:
                break
            trial = pair(base, label + '_Ponly', start, roles=('prefill',))
            trials.append(trial)
            write(target, dict(status='IN_PROGRESS', results=results, current=trials))
            if not trial['passed']:
                break
        results.append(dict(candidate=candidate, trials=trials))
        if len(trials) == 3 and all(row['passed'] for row in trials):
            write(ROOT / 'state/huginn_final_recovery_gate.json', dict(
                status='RECOVERY_READY', same_run_gates=True, whole_run_decode_SLO=True,
                independent_repeats=True, exact_state_and_output=True,
                qualified_configuration=base, geometry_state_gate=candidate, results=trials,
                D_kernel_activity='NOT_MEASURED_IN_ACCEPTED_RUNS',
                scope='Original operational Phase A3 gates. Exact D work, credits, and SLO come from each accepted P-profiled execution.'))
            write(target, dict(status='RECOVERY_READY', results=results))
            from .conditional_split_qualification import main as qualify
            from .static_screen import main as screen
            qualify()
            screen()
            return
    write(target, dict(status='RECOVERY_FAILED_OR_INCOMPLETE', results=results))


if __name__ == '__main__':
    main()
