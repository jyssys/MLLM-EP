"""Finish the original clean/profile differential before changing variables."""
import time
from .common import *
from .run import execute


def main():
    initialize()
    clean = 'recovery_huginn_original_clean384'
    result = ROOT / 'raw' / clean / 'dispatcher_result.json'
    while not result.exists():
        if prior.time.time() >= LONG_START_CUTOFF:
            return
        time.sleep(10)
    assert json.loads(result.read_text())['complete'], 'Clean reproduction failed'
    name = 'recovery_huginn_original_profile384'
    complete = execute(name)
    folder = ROOT / 'raw' / name
    watchdog = folder / 'watchdog.json'
    reason = json.loads(watchdog.read_text()).get('reason') if watchdog.exists() else None
    stacks = '\n'.join(path.read_text() for path in folder.glob('worker_stacks_*.log'))
    exact_symptom = (reason == 'PROFILE_NO_RESPONSE_PROGRESS_60S'
                     and 'nixl_cu12/_api.py' in stacks and 'in transfer' in stacks)
    write(ROOT / 'state/reproduction_result.json', dict(clean=clean, profile=name,
        profile_complete=complete, time=time.time(),
        watchdog_reason=reason, exact_previous_stall_reproduced=exact_symptom,
        status='REVIEW_PROGRESS_AND_PARITY' if complete else
               'RED_REPRODUCER' if exact_symptom else 'OTHER_FAILURE_REVIEW_REQUIRED'))


if __name__ == '__main__':
    main()
