"""Minimize the observed NIXL progress stall with owned native stack capture."""
import argparse
import copy
from .common import *
from .run import execute
from .recovery_qualification import save


def main(count):
    initialize()
    original = ROOT / 'raw/recovery_huginn_extended1_profile512'
    result = json.loads((original / 'dispatcher_result.json').read_text())
    watchdog = json.loads((original / 'watchdog.json').read_text())
    stacks = '\n'.join(p.read_text() for p in original.glob('worker_stacks_*.log'))
    assert not result['complete'] and 'in transfer' in stacks and 'nixl_cu12/_api.py' in stacks
    write(ROOT / 'state/extended_stall_reproduced.json', dict(
        exact_progress_stall_reproduced=True, source=str(original),
        watchdog=watchdog, limitation='Python location is not the native root cause.'))
    cfg = copy.deepcopy(json.loads((ROOT / 'configs/recovery_huginn_extended1_profile512.json').read_text()))
    cfg.update(name=f'recovery_huginn_stall_native{count}', count=count,
               native_stack_diagnostic=True, phase='NIXL_STALL_DIAGNOSTIC_NOT_CLEAN_TIMING')
    for key in ('request_caps', 'prompt_ids', 'arrival_offsets'):
        cfg[key] = cfg[key][:count]
    cfg.pop('matched_clean', None)
    save(cfg)
    complete = execute(cfg['name'])
    write(ROOT / 'state' / f'stall_native{count}.json', dict(name=cfg['name'], complete=complete,
        change='Cohort truncation only. Native debugger attaches after 30 seconds without responses.'))


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--count', type=int, default=128)
    main(p.parse_args().count)
