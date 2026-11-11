"""Copy immutable anchor configurations without changing frozen evidence."""
import copy
import shutil
from .common import *


def clone(source, name, **updates):
    cfg = json.loads((PRIOR_ROOT / 'configs' / (source + '.json')).read_text())
    cfg.update(name=name, campaign='huginn_native_relative_net_benefit', **updates)
    cfg.pop('required_qualification', None)
    cfg.pop('primary_safety_arm', None)
    cfg['net_benefit_parent_configuration'] = dict(path=str(PRIOR_ROOT / 'configs' / (source + '.json')),
        sha256=digest(PRIOR_ROOT / 'configs' / (source + '.json')))
    return cfg


def main():
    initialize()
    for source in (PRIOR_ROOT / 'data').glob('*huginn*.json'):
        destination = ROOT / 'data' / source.name
        if not destination.exists():
            shutil.copy2(source, destination)
    for filename in ('frozen_SLO.json',):
        destination = ROOT / 'configs' / filename
        if not destination.exists():
            shutil.copy2(PRIOR_ROOT / 'configs' / filename, destination)
    # Inputs remain the exact preserved tokenized pool. Absolute paths are valid.
    jobs = []
    for suffix in ('a', 'b'):
        name = 'baseline_zero_seed2026110931_' + suffix
        cfg = clone('control_huginn_zero_seed2026110931', name,
            phase='FRESH_STRONG_BASELINE_AA', clean_confirmation=True, timeout=600)
        save(cfg)
        jobs.append(name)
    write(ROOT / 'state/baseline_jobs.json', jobs)
    print(' '.join(jobs))


if __name__ == '__main__':
    main()
