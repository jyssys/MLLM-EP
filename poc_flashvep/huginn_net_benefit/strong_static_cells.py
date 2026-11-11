"""Give large Token suffixes and deeper Loop cuts the same qualification budget."""
from .common import *


def main():
    initialize()
    jobs = []
    variants = (
        ('token1792', 'quantum6_token1984_1', dict(native_frontier=[0,1792])),
        ('token1536', 'quantum6_token1984_1', dict(native_frontier=[0,1536])),
        ('loop28_full', 'priority4_normal_loop31', dict(native_partial_cut=28, cut=28,
            D_prefill_loop_cap=1)),
        ('loop28_tile128', 'quantum6_loop31_128', dict(native_partial_cut=28, cut=28)))
    for arm, source, options in variants:
        for suffix in ('state8', 'state32', 'seed2026110940'):
            cfg = json.loads((ROOT/'configs'/f'{source}_{suffix}.json').read_text())
            cfg.update(name=f'static11_{arm}_{suffix}',
                phase='MATCHED_LARGE_TOKEN_AND_DEEP_LOOP_CUT_QUALIFICATION_OR_SELECTION', **options)
            if suffix == 'state32':
                cfg['required_qualification'] = f'static11_{arm}_state8'
            if suffix.startswith('seed'):
                cfg['required_qualification'] = f'static11_{arm}_state32'
            save(cfg)
            jobs.append(cfg['name'])
    write(ROOT/'state/strong_static_comparison_jobs.json', jobs)
    print(' '.join(jobs))


if __name__ == '__main__':
    main()
