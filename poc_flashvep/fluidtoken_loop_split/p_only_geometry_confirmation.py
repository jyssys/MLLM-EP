"""Retain the original P-trace gate when D tracing invalidates Decode SLO.

D evidence still comes from exact real probes, credits, and latency in the
same P-profiled execution. D kernel intervals remain explicitly unavailable.
"""
import time
from .common import *
from .geometry_confirmation import pair


def main():
    initialize()
    pending = ROOT / 'state/geometry_confirmation.json'
    target = ROOT / 'state/p_only_geometry_confirmation.json'
    write(ROOT / 'state/p_only_geometry_plan.json', dict(
        reason='Dual-device profiling changed Decode tails although total wall time changed by less than 5%.',
        trigger='Run only if dual-device confirmation fails.',
        unchanged_gates=['P kernel union >=80%', 'P finite service proximity >=85%',
            'Positive device-qualified queue growth', 'Profile wall difference <=5%',
            'D credit gate and refusal fraction <=50%', 'Frozen same-window Decode SLO',
            'Whole-run Decode SLO for clean and profiled runs', 'Exact real D probes and full work',
            'Exact native background outputs', 'Three independent starts'],
        scope='Original Phase A3 requires P kernel activity plus same-run D probes, KV, and Decode SLO. It does not require a D kernel-union threshold.',
        limitation='Accepted P-only runs cannot provide measured D kernel timelines. Failed dual-device traces cannot fill that panel.',
        order=['slots4_graph4', 'slots8_graph8'], starts=[1, 2, 3]))
    while time.time() < LONG_START_CUTOFF:
        if pending.exists():
            previous = json.loads(pending.read_text())
            if previous['status'] in ('RECOVERY_READY', 'RECOVERY_FAILED', 'NOT_RUN_TIME_BUDGET'):
                break
        time.sleep(5)
    else:
        write(target, dict(status='NOT_RUN_TIME_BUDGET'))
        return
    if previous['status'] == 'RECOVERY_READY':
        write(target, dict(status='NOT_NEEDED_DUAL_GATE_PASSED'))
        return
    state = json.loads((ROOT / 'state/tail_geometry_control.json').read_text())
    results = []
    for candidate in state['results']:
        if not candidate.get('passed') or candidate['label'] == 'native8_graph4':
            continue
        base = json.loads((ROOT / 'configs' / (candidate['name'] + '.json')).read_text())
        trials = []
        for start in (1, 2, 3):
            if time.time() >= LONG_START_CUTOFF:
                break
            trial = pair(base, candidate['label'] + '_Ponly', start, roles=('prefill',))
            trials.append(trial)
            write(target, dict(status='IN_PROGRESS', completed_candidates=results, current=trials))
            if not trial['passed']:
                break
        results.append(dict(candidate=candidate, trials=trials))
        if len(trials) == 3 and all(row['passed'] for row in trials):
            write(ROOT / 'state/huginn_final_recovery_gate.json', dict(
                status='RECOVERY_READY', same_run_gates=True, whole_run_decode_SLO=True,
                independent_repeats=True, exact_state_and_output=True,
                qualified_configuration=base, geometry_state_gate=candidate, results=trials,
                D_kernel_activity='NOT_MEASURED_IN_ACCEPTED_RUNS',
                dual_device_limitation='Retained dual-device captures failed. Do not combine their D timeline with these accepted runs.',
                scope='Original operational Phase A3 gates, with real same-run D probe and SLO evidence. Not a peak-FLOP or migrated-work claim.'))
            write(target, dict(status='RECOVERY_READY', results=results))
            return
    write(target, dict(status='RECOVERY_FAILED', results=results))
    write(ROOT / 'state/huginn_final_recovery_gate.json', dict(status='RECOVERY_FAILED', results=results))


if __name__ == '__main__':
    main()
