"""Confirm the fixed admitted-request denominator with new independent starts."""
import time
from .common import *
from .geometry_confirmation import pair


def main():
    initialize()
    target = ROOT / 'state/admitted_probe_confirmation.json'
    write(ROOT / 'state/admitted_probe_confirmation_plan.json', dict(
        reason='D_probe_count is an admission upper limit. The prior checker incorrectly required that limit to be reached.',
        actual_case='The slots8 all-phase profile admitted six probes and completed all six. The prior quota verdict remains false.',
        contract='Require at least one exact completed probe in the common interval and all admitted probes completed exactly. Keep full background completion, all numerical checks, and every frozen performance threshold.',
        validation='Three new independent starts. Do not use the exploratory run as a confirmation start.',
        upper_limit=12, count=512, starts=[1, 2, 3], D_slots=8, graph_rows=8, QPS=1.971))
    write(target, dict(status='WAITING_FOR_OLD_CONTRACT_CONTROL'))
    pending = ROOT / 'state/all_phase_confirmation.json'
    while time.time() < LONG_START_CUTOFF:
        if pending.exists():
            state = json.loads(pending.read_text())
            if state['status'] in ('RECOVERY_READY', 'NOT_NEEDED'):
                write(target, dict(status='NOT_NEEDED'))
                return
            if state['status'] == 'RECOVERY_FAILED_OR_INCOMPLETE':
                break
        time.sleep(5)
    else:
        write(target, dict(status='NOT_RUN_TIME_BUDGET'))
        return
    base = json.loads((ROOT / 'configs/recovery_huginn_allphase_slots8_q1p971_joint_s1_clean512.json').read_text())
    base['D_probe_completion_contract'] = 'admitted_exact_v2'
    from .probe_reference import verify_admitted
    from ..looped_pd_autonomous_headroom.analysis_lane import quiet_analysis
    with quiet_analysis('probe_admission_denominator_audit'):
        exploratory = verify_admitted('recovery_huginn_allphase_slots8_q1p971_joint_s1_profile512')
    results = []
    for start in (1, 2, 3):
        if time.time() >= LONG_START_CUTOFF:
            break
        write(target, dict(status='IN_PROGRESS', current_start=start,
            exploratory_review=exploratory, results=results))
        result = pair(base, 'allphase_slots8_admitted_v2', start, roles=('prefill',))
        results.append(result)
        write(target, dict(status='IN_PROGRESS', exploratory_review=exploratory, results=results))
        if not result['passed']:
            break
    if len(results) == 3 and all(row['passed'] for row in results):
        previous = json.loads((ROOT / 'state/evidence_routing_control.json').read_text())
        geometry = next(row for row in previous['results'] if row['label'] == 'async_evidence_slots8_graph8')
        write(ROOT / 'state/huginn_final_recovery_gate.json', dict(
            status='RECOVERY_READY', same_run_gates=True, whole_run_decode_SLO=True,
            independent_repeats=True, exact_state_and_output=True,
            qualified_configuration=base, geometry_state_gate=geometry, results=results,
            D_kernel_activity='NOT_MEASURED_IN_ACCEPTED_RUNS',
            probe_completion_contract='ADMITTED_EXACT_V2',
            scope='All admitted probes and background requests finish exactly. The configured probe count remains a maximum admission count.'))
        write(target, dict(status='RECOVERY_READY', exploratory_review=exploratory, results=results))
        from .conditional_split_qualification import main as qualify
        from .static_screen import main as screen
        qualify()
        screen()
        return
    write(target, dict(status='RECOVERY_FAILED_OR_INCOMPLETE', exploratory_review=exploratory, results=results))


if __name__ == '__main__':
    main()
