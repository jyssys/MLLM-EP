"""Tune both axes on discovery data before selecting held-out arms."""
import fcntl
import json
import time
from .common import ROOT, QUIET_LOCK, resources, save, write
from .axis_configuration import base
from .analysis import metrics
from .run import execute


def job(cfg):
    complete = execute(save(cfg))
    if not complete:
        write(ROOT / 'state/discovery_failure_' / (cfg['name'] + '.json'),
              dict(status='FAILED', time=time.time(), name=cfg['name']))
        return None
    result = metrics(cfg['name'])
    assert result['output_gate']['passed'], 'Original output mismatch: ' + cfg['name']
    return result


def main():
    assert json.loads((ROOT / 'analysis/Layer_all_cuts_N32_full_state_gate.json').read_text())['passed']
    assert json.loads((ROOT / 'analysis/Token2048_state_N32_O16_v1_state_gate.json').read_text())['passed']
    jobs = []
    for method in ('Native', 'PChunk', 'Token', 'Static', 'Dynamic'):
        candidates = (512, 1024, 2048, 4096) if method == 'Token' else (2, 4, 8, 12) if method == 'Static' else (0,)
        for candidate in candidates:
            name = f'discovery_{method}_{candidate}_N64_O16_q4.8_s2026111710_v1'
            jobs.append(base(name, method=method, layers=candidate if method == 'Static' else 8,
                suffix=candidate if method == 'Token' else 512, count=64, cap=16, rate=4.8,
                D_combine_ready_macros=True, D_wait_for_source_macro=False))
    try:
        values = []
        for cfg in jobs:
            value = job(cfg)
            if value is not None:
                values.append(value)
                write(ROOT / 'analysis/discovery_O16_first_grid.json', values)
        write(ROOT / 'state/discovery_O16_first_grid_complete.json', dict(time=time.time(),
            completed=[v['name'] for v in values], expected=[c['name'] for c in jobs]))
    finally:
        with QUIET_LOCK.open('a+') as lease:
            fcntl.flock(lease, fcntl.LOCK_EX)
            if not resources().processes():
                resources().restore(list(range(8)))


if __name__ == '__main__':
    main()
