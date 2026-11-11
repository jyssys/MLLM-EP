"""Screen qualified cuts before a long serving cohort. Keep every outcome."""
import argparse
from .common import *


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('selection_configs', nargs='+')
    args = parser.parse_args()
    initialize()
    jobs = []
    for source in args.selection_configs:
        path = ROOT/'configs'/(source+'.json')
        cfg = json.loads(path.read_text())
        assert cfg['count'] == 128 and not cfg.get('state_probe')
        cfg.update(name='screen_'+source, count=32, request_caps=[cfg['cap']]*32,
            arrival_offsets=cfg['arrival_offsets'][:32],
            phase='N32_CLEAN_CUT_SCREEN_NOT_CONFIRMATION',
            full_selection_configuration=str(path), full_selection_sha256=digest(path),
            screen_rule='Retain every N32 result. Expand promising cuts only after exact-state qualification. A finite short cohort does not measure sustainable capacity.')
        save(cfg)
        jobs.append(cfg['name'])
    print(' '.join(jobs))


if __name__ == '__main__':
    main()
