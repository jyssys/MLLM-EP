"""Preserve the failed-latch control and qualify one drained-epoch retry."""
from .common import *


def main():
    initialize()
    jobs = []
    for arm in ('loop', 'token'):
        label = f'retry17_output4_{arm}'
        for suffix in ('state8', 'state32', 'seed2026110940'):
            source = ROOT/'configs'/f'forecast8_output4_{arm}_{suffix}.json'
            cfg = json.loads(source.read_text())
            name = f'{label}_{suffix}'
            cfg.update(name=name, phase='SHORT_OUTPUT_DRAINED_RETRY_NOT_PRIMARY_ANCHOR',
                drained_completion_recovery=True, failed_latch_parent_sha256=digest(source))
            if suffix != 'state8':
                cfg['required_qualification'] = label+('_state8' if suffix == 'state32' else '_state32')
            save(cfg)
            jobs.append(name)
    write(ROOT/'state/drained_completion_retry_jobs.json', jobs)
    print(' '.join(jobs))


if __name__ == '__main__':
    main()
