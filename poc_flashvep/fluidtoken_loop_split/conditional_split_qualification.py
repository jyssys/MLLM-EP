"""Keep method GPU work behind the independently confirmed Huginn gate."""
import time
from .common import *


def recovery_ready(state):
    excluded = {json.loads(path.read_text())['run']
                for path in (ROOT / 'state').glob('host_audit_overlap_*.json')
                if json.loads(path.read_text()).get('exclude_from_clean_confirmation')}
    return (state.get('status') == 'RECOVERY_READY' and all(state.get(key, False) for key in
        ('same_run_gates', 'whole_run_decode_SLO', 'independent_repeats', 'exact_state_and_output'))
        and not any(row.get('profile') in excluded or row.get('clean') in excluded
                    for row in state.get('results', [])))


def main():
    initialize()
    # Bounded-window passes did not guarantee whole-run Decode SLO compliance.
    # Do not unlock method work with the earlier, narrower recovery artifact.
    path = ROOT / 'state/huginn_final_recovery_gate.json'
    terminal = {'RECOVERY_READY', 'BLOCKED_CORRECTNESS', 'RECOVERY_FAILED'}
    while time.time() < LONG_START_CUTOFF:
        if path.exists():
            state = json.loads(path.read_text())
            if state['status'] in terminal:
                break
        time.sleep(5)
    else:
        write(ROOT / 'state/conditional_split_qualification.json', dict(status='NOT_RUN_TIME_BUDGET'))
        return
    if not recovery_ready(state):
        write(ROOT / 'state/conditional_split_qualification.json', dict(
            status='BLOCKED_BY_HUGINN_GATE', recovery_status=state['status']))
        return
    from .matched_no_probe import run as matched_control
    write(ROOT / 'state/conditional_split_qualification.json', dict(
        status='IN_PROGRESS', phase='MATCHED_NO_PROBE_CONTROL'))
    if not matched_control(state['qualified_configuration']):
        write(ROOT / 'state/conditional_split_qualification.json', dict(
            status='BLOCKED_MATCHED_CONTROL_CORRECTNESS'))
        return
    from .static_qualification import main as qualify, extended
    results = {}
    extended_results = {}
    for model in ('huginn', 'ouro'):
        write(ROOT / 'state/conditional_split_qualification.json', dict(
            status='IN_PROGRESS', phase='CONNECTOR_STATE_QUALIFICATION', model=model,
            results=results, extended_results=extended_results))
        qualify(model, True)
        results[model] = json.loads((ROOT / 'state' / f'static_qualification_{model}.json').read_text())
        extended_results[model] = extended(model)
        write(ROOT / 'state/conditional_split_qualification.json', dict(status='IN_PROGRESS', results=results, extended_results=extended_results))
    write(ROOT / 'state/conditional_split_qualification.json', dict(status='QUALIFICATION_FINISHED', results=results, extended_results=extended_results))


if __name__ == '__main__':
    main()
