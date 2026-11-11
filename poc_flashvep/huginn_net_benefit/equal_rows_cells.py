"""Compare a qualified 256-token suffix with four complete recurrent Loops."""
from .common import *


def main():
    initialize()
    jobs = []
    for label, parent in (('match18_loop28_full', 'priority4_normal_loop31'),
                          ('match18_loop28_tile256', 'quantum6_loop31_256')):
        for suffix in ('state8', 'state32', 'seed2026110940'):
            source = ROOT/'configs'/(parent+'_'+suffix+'.json')
            cfg = json.loads(source.read_text())
            name = label+'_'+suffix
            cfg.update(name=name, phase='EQUAL_RECURRENT_ROWS_FIXED_CUT_DIAGNOSTIC',
                native_partial_cut=28, cut=28, D_prefill_loop_cap=1,
                require_effective_quantum_qualification=True,
                matched_token_rows_config='token16_prefix1792_cap8_'+suffix,
                fixed_frontier_parent_sha256=digest(source))
            if suffix != 'state8':
                cfg['required_qualification'] = label+('_state8' if suffix == 'state32' else '_state32')
            save(cfg)
            jobs.append(name)
    write(ROOT/'state/equal_semantic_rows_jobs.json', jobs)
    print(' '.join(jobs))


if __name__ == '__main__':
    main()
