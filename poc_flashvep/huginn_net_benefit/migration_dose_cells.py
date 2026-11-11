"""Prepare a fixed, balanced placement diagnostic, not an adaptive policy."""
import argparse
import random
from .common import *


def prepare(source, stride, assignment_seed):
    initialize()
    assert stride in (2, 4, 8)
    jobs = []
    label = f'dose14_every{stride}'
    for suffix in ('state8', 'state32', 'seed2026110940'):
        original = ROOT/'configs'/(source+'_'+suffix+'.json')
        cfg = json.loads(original.read_text())
        assert cfg['model'] == 'huginn' and cfg['depth'] == 32
        assert cfg['native_partial_cut'] == 31 and cfg.get('native_frontier') is None
        assert cfg['count'] % stride == 0
        rng = random.Random(assignment_seed)
        cuts = [32]*cfg['count']
        for start in range(0, cfg['count'], stride):
            cuts[start+rng.randrange(stride)] = 31
        name = f'{label}_{suffix}'
        cfg.update(name=name, phase='FORCED_FIXED_DOSE_DIAGNOSTIC_NOT_ADAPTIVE',
            native_request_cuts=cuts, placement_seed=assignment_seed,
            placement_selected_requests=cfg['count']//stride,
            placement_source_sha256=digest(original),
            forced_cut_scope='One placement in each contiguous request-index block. Assignment uses no future timing or measured outcomes. The cut is fixed at reservation and is not an online Slack decision.')
        if suffix != 'state8':
            cfg['required_qualification'] = label+('_state8' if suffix == 'state32' else '_state32')
        save(cfg)
        jobs.append(name)
    write(ROOT/'state'/(label+'_jobs.json'), jobs)
    return jobs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('source', help='Qualified fixed Loop31 family without its cell suffix')
    parser.add_argument('--stride', type=int, default=4)
    parser.add_argument('--assignment-seed', type=int, default=2026111073)
    args = parser.parse_args()
    print(' '.join(prepare(args.source, args.stride, args.assignment_seed)))


if __name__ == '__main__':
    main()
