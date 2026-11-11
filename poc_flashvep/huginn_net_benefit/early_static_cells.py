"""Pair early wire overlap with a fixed exact cut and an equivalent Zero."""
from .common import *


def main():
    initialize()
    jobs = []
    label = 'early15_output16_loop31'
    for suffix in ('state8', 'state32', 'seed2026110940'):
        source = ROOT/'configs'/('early9_output16_zero_'+suffix+'.json')
        cfg = json.loads(source.read_text())
        name = label+'_'+suffix
        cfg.update(name=name, phase='EARLY_WIRE_FIXED_CUT_INTERVENTION',
            native_partial_cut=31, cut=31, method='loop', axis='loop',
            atomic_prompt=False, forced_offload_diagnostic=True,
            matched_early_zero=source.stem, matched_early_zero_configuration_sha256=digest(source))
        if suffix != 'state8':
            cfg['required_qualification'] = label+('_state8' if suffix == 'state32' else '_state32')
        save(cfg)
        jobs.append(name)
    write(ROOT/'state/early_static_overlap_jobs.json', jobs)
    print(' '.join(jobs))


if __name__ == '__main__':
    main()
