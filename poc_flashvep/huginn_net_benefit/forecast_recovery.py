"""Preserve the failed flag check and retry the unchanged model path separately."""
from .common import *


def main():
    initialize()
    jobs = []
    for suffix in ('state8', 'state32', 'seed2026110940'):
        source = f'forecast8_output16_loop_{suffix}'
        cfg = json.loads((ROOT/'configs'/(source+'.json')).read_text())
        cfg.update(name=f'forecast8b_output16_loop_{suffix}',
            recovery_scope='The parent Loop family uses the dynamic_tile enum. The corrected optional bootstrap accepts that existing Loop-frontier path. Model, cuts, budgets, and Native validation remain unchanged.')
        if suffix == 'state32':
            cfg['required_qualification'] = 'forecast8b_output16_loop_state8'
        if suffix.startswith('seed'):
            cfg['required_qualification'] = 'forecast8b_output16_loop_state32'
        save(cfg)
        jobs.append(cfg['name'])
    print(' '.join(jobs))


if __name__ == '__main__':
    main()
