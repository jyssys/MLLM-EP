"""Collect native stacks in a shorter reproduction, not a clean timing run."""
import time
from .common import *
from .run import execute


def main():
    initialize()
    original = ROOT / 'raw/recovery_huginn_original_profile384'
    while not (original / 'dispatcher_result.json').exists():
        if time.time() >= LONG_START_CUTOFF:
            return
        time.sleep(5)
    result = json.loads((original / 'dispatcher_result.json').read_text())
    watchdog = original / 'watchdog.json'
    reason = json.loads(watchdog.read_text()).get('reason') if watchdog.exists() else None
    stacks = '\n'.join(path.read_text() for path in original.glob('worker_stacks_*.log'))
    red = (not result['complete'] and reason == 'PROFILE_NO_RESPONSE_PROGRESS_60S'
           and 'nixl_cu12/_api.py' in stacks and 'in transfer' in stacks)
    write(ROOT / 'state/reproduction_result_audited.json', dict(
        exact_previous_stall_reproduced=red, dispatcher=result, watchdog_reason=reason))
    if not red:
        return
    cfg = json.loads((ROOT / 'configs/recovery_huginn_original_profile384.json').read_text())
    cfg.update(name='recovery_huginn_minimized_profile128_native_stack', count=128,
               native_stack_diagnostic=True, phase='DIAGNOSTIC_NOT_CLEAN_TIMING')
    for key in ('request_caps', 'prompt_ids', 'arrival_offsets'):
        cfg[key] = cfg[key][:128]
    write(ROOT / 'configs' / (cfg['name'] + '.json'), cfg)
    complete = execute(cfg['name'])
    write(ROOT / 'state/minimized_result.json', dict(name=cfg['name'], complete=complete,
        scope='Cohort reduced to first 128. Capture remains at 35% of that arrival span. Native attachment occurs only after progress stops.'))


if __name__ == '__main__':
    main()
