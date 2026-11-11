"""An actual disjoint allocation is not the earlier prompt-only SM cap."""
from .common import *


def main():
    initialize()
    jobs = []
    for arm in ('zero', 'loop31', 'token1984'):
        for suffix in ('state8', 'state32', 'seed2026110940'):
            cfg = json.loads((ROOT/'configs'/f'priority4_normal_{arm}_{suffix}.json').read_text())
            cfg.update(name=f'disjoint13_sm8_{arm}_{suffix}',
                disjoint_green_prompt_sms=8, cuda_device_max_connections=32,
                phase='DISJOINT_DOCUMENTED_RESOURCE_QUALIFICATION_OR_INTERVENTION',
                spatial_scope='Decode and prompt resources come from disjoint groups of one CUDA resource split. This is not a kernel-overlap or safety guarantee. Compare against the strongest full-device Zero too.')
            if suffix == 'state32':
                cfg['required_qualification'] = f'disjoint13_sm8_{arm}_state8'
            if suffix.startswith('seed'):
                cfg['required_qualification'] = f'disjoint13_sm8_{arm}_state32'
            save(cfg)
            jobs.append(cfg['name'])
    write(ROOT/'state/disjoint_SM_intervention_jobs.json', jobs)
    print(' '.join(jobs))


if __name__ == '__main__':
    main()
