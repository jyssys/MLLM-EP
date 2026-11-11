"""Requalify Token suffix grouping after observing an uncapped Prelude."""
from .common import *


def main():
    initialize()
    jobs = []
    for prefix, cap in ((1984, 8), (1792, 8), (1984, 1)):
        label = f'token16_prefix{prefix}_cap{cap}'
        for suffix in ('state8', 'state32', 'seed2026110940'):
            source = ROOT/'configs'/('quantum6_token1984_1_'+suffix+'.json')
            cfg = json.loads(source.read_text())
            name = label+'_'+suffix
            cfg.update(name=name, phase='OBSERVED_TOKEN_EXECUTION_QUANTUM_QUALIFICATION',
                native_frontier=[0,prefix], D_prefill_loop_cap=cap,
                D_prefill_cap_includes_prelude=True, require_effective_quantum_qualification=True,
                unbounded_prelude_configuration_sha256=digest(source))
            if suffix != 'state8':
                cfg['required_qualification'] = label+('_state8' if suffix == 'state32' else '_state32')
            save(cfg)
            jobs.append(name)
    write(ROOT/'state/corrected_token_quantum_jobs.json', jobs)
    print(' '.join(jobs))


if __name__ == '__main__':
    main()
