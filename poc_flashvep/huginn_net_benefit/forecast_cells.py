"""Test preparatory forecasts without modifying boundary admission budgets."""
from .common import *


def main():
    initialize()
    jobs = []
    for cap in (16, 4):
        for arm in ('loop', 'token'):
            label = f'forecast8_output{cap}_{arm}'
            for suffix in ('state8', 'state32', 'seed2026110940'):
                cfg = json.loads((ROOT/'configs'/f'pipeline7_output{cap}_{arm}_{suffix}.json').read_text())
                cfg.update(name=f'{label}_{suffix}', predictive_checkpoint_preparation=True,
                    preparation_decode_cycle_ms=93.10616207122803,
                    preparation_source_prompt_ms=600.71,
                    phase='KNOWN_WORK_PREPARATION_FORECAST_NOT_COMPLETION_GUARANTEE')
                if suffix == 'state32':
                    cfg['required_qualification'] = f'{label}_state8'
                if suffix.startswith('seed'):
                    cfg['required_qualification'] = f'{label}_state32'
                save(cfg)
                jobs.append(cfg['name'])
    write(ROOT/'state/predictive_preparation_jobs.json', jobs)
    print(' '.join(jobs))


if __name__ == '__main__':
    main()
