"""Preserve failed namespaces and resolve the common qualification frontier."""
from .common import *


def main():
    initialize()
    names = []
    for arm in ('sourceevents', 'fastloop'):
        for count in (8, 32):
            cfg = json.loads((ROOT/'configs'/f'candidate_{arm}_state{count}.json').read_text())
            cfg.update(name=f'candidate_{arm}_v2_state{count}',
                boundary_checkpoints=[16, 21, 24, 28, 30, 31])
            if count == 32:
                cfg['required_qualification'] = f'candidate_{arm}_v2_state8'
            save(cfg)
            names.append(cfg['name'])
    cfg = json.loads((ROOT/'configs/selection_fastloop_seed2026110940.json').read_text())
    cfg.update(name='selection_fastloop_v2_seed2026110940',
        required_qualification='candidate_fastloop_v2_state32')
    save(cfg)
    names.append(cfg['name'])
    # Discovery configurations also use this explicitly qualified candidate set.
    for count in (8, 32):
        original = ROOT/'configs'/f'discovery_output4_loop_state{count}.json'
        cfg = json.loads(original.read_text())
        cfg.update(name=f'discovery_output4_loop_v2_state{count}',
            boundary_checkpoints=[16, 21, 24, 28, 30, 31])
        if count == 32:
            cfg['required_qualification'] = 'discovery_output4_loop_v2_state8'
        save(cfg)
    cfg = json.loads((ROOT/'configs/discovery_output4_loop_seed2026110940.json').read_text())
    cfg.update(name='discovery_output4_loop_v2_seed2026110940',
        required_qualification='discovery_output4_loop_v2_state32')
    save(cfg)
    write(ROOT/'state/frontier_repair_jobs.json', names)
    print(' '.join(names))


if __name__ == '__main__':
    main()
