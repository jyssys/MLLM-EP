"""Replace a serving repetition that overlapped an unguarded repository audit."""
import time
from .common import *


def main():
    initialize()
    target = ROOT / 'state/audit_replacement_confirmation.json'
    path = ROOT / 'state/admitted_probe_confirmation.json'
    write(target, dict(status='WAITING', reason='Do not use the host-audited first profile as clean confirmation.'))
    while time.time() < LONG_START_CUTOFF:
        state = json.loads(path.read_text())
        if state['status'] in ('RECOVERY_FAILED_OR_INCOMPLETE', 'NOT_RUN_TIME_BUDGET', 'NOT_NEEDED'):
            write(target, dict(status='NOT_NEEDED', previous=state['status']))
            return
        if state['status'] == 'RECOVERY_READY':
            break
        time.sleep(5)
    else:
        write(target, dict(status='NOT_RUN_TIME_BUDGET'))
        return
    gate_path = ROOT / 'state/huginn_final_recovery_gate.json'
    gate = json.loads(gate_path.read_text())
    write(ROOT / 'state/pre_audit_replacement_recovery_gate.json', gate)
    from .geometry_confirmation import pair
    result = pair(gate['qualified_configuration'], 'allphase_slots8_audit_replacement', 4, roles=('prefill',))
    if not result['passed']:
        write(target, dict(status='RECOVERY_FAILED', result=result))
        return
    gate['results'] = gate['results'][1:] + [result]
    gate['excluded_confirmation_start'] = 1
    gate['replacement_independent_start'] = 4
    write(gate_path, gate)
    from .conditional_split_qualification import recovery_ready, main as qualify
    assert recovery_ready(gate)
    write(target, dict(status='RECOVERY_READY', result=result))
    qualify()
    from .static_screen import main as screen
    screen()


if __name__ == '__main__':
    main()
