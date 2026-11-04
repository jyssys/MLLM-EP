"""Prepare an exact old failure and its clean control in new namespaces."""
import copy
import shutil
from .common import *


def main():
    initialize()
    for path in (PREVIOUS / 'data').glob('*.json'):
        target = ROOT / 'data' / path.name
        if not target.exists():
            shutil.copyfile(path, target)
    target = ROOT / 'configs/frozen_SLO.json'
    if not target.exists():
        shutil.copyfile(PREVIOUS / 'configs/frozen_SLO.json', target)
    names = []
    for label, old in [('clean', 'short_huginn_s1_probe384'),
                       ('profile', 'short_huginn_s1_nsys384')]:
        cfg = json.loads((PREVIOUS / 'configs' / (old + '.json')).read_text())
        cfg.update(name='recovery_huginn_original_' + label + '384',
                   phase='EXACT_PREVIOUS_FAILURE_REPRODUCTION')
        if label == 'profile':
            cfg['matched_clean'] = names[0]
        path = ROOT / 'configs' / (cfg['name'] + '.json')
        if path.exists():
            assert json.loads(path.read_text()) == cfg
        else:
            write(path, cfg)
        names.append(cfg['name'])
    write(ROOT / 'state/reproduction_plan.json', dict(names=names,
        scope='Same data, arrivals, geometry, and watchdog as previous failure. New output namespace only.',
        inherited_SLO_SHA256=digest(target)))
    print(json.dumps(names))


if __name__ == '__main__':
    main()
