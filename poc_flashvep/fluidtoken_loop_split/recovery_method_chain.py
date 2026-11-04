"""Continue exact method tests only after the current recovery control passes."""
import time
from .common import *


def main():
    initialize()
    source = ROOT / 'state/evidence_routing_control.json'
    target = ROOT / 'state/recovery_method_chain.json'
    write(target, dict(status='WAITING_FOR_CURRENT_RECOVERY'))
    while time.time() < LONG_START_CUTOFF:
        if source.exists():
            result = json.loads(source.read_text())
            if result['status'] == 'RECOVERY_READY':
                break
            if result['status'] in ('RECOVERY_FAILED_OR_INCOMPLETE', 'NO_MEASURED_IO_PAUSE', 'DIAGNOSTIC_INCOMPLETE'):
                write(target, dict(status='BLOCKED_BY_HUGINN_GATE', recovery_status=result['status']))
                return
        time.sleep(5)
    else:
        write(target, dict(status='NOT_RUN_TIME_BUDGET'))
        return
    from .conditional_split_qualification import main as qualify
    from .static_screen import main as screen
    write(target, dict(status='EXACT_CONNECTOR_QUALIFICATION'))
    qualify()
    write(target, dict(status='CONDITIONAL_STATIC_SCREEN'))
    screen()
    write(target, dict(status='STATIC_STAGE_FINISHED_REVIEW_REQUIRED'))


if __name__ == '__main__':
    main()
