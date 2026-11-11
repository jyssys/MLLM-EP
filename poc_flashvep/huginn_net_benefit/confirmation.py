"""Freeze selected arms before balanced, held-out Arrival-seed confirmation."""
import argparse
import random
from .common import *


def arrivals(seed, count, qps):
    rng = random.Random(seed)
    offsets = [0.]
    for _ in range(count-1):
        offsets.append(offsets[-1]+rng.expovariate(qps))
    return offsets


def prepare(arms, seeds, tag, cap=16):
    initialize()
    assert len(arms) >= 2 and arms[0][0] == 'zero'
    assert all(label.isalnum() and label.islower() for label,_ in arms), (
        'Start-level pairing requires one lowercase alphanumeric label token')
    assert len({label for label,_ in arms}) == len(arms)
    assert len(set(seeds)) == len(seeds)
    assert 2026110940 not in seeds, 'The selection trace is not held out'
    names, cells = [], {}
    frozen = time.time()
    for label, source in arms:
        path = ROOT/'configs'/(source+'.json')
        original = json.loads(path.read_text())
        assert cap in (4, 16)
        assert original['count'] == 128 and original['length'] == 2048 and original['cap'] == cap
        assert not original.get('state_probe')
        for seed in seeds:
            cfg = dict(original)
            name = f'{tag}_{label}_seed{seed}'
            cfg.update(name=name, phase='HELD_OUT_BALANCED_CONFIRMATION',
                arrival_seed=seed, arrival_offsets=arrivals(seed, 128, 1.971),
                selection_configuration=str(path), selection_sha256=digest(path),
                selection_frozen_epoch=frozen,
                confirmation_label=label, clean_confirmation=True)
            save(cfg)
            cells[label, seed] = name
    # Each seed has a second independent Zero start. The sequence rotates
    # candidate order. Requests share only their explicit paired arrival trace.
    labels = [label for label, _ in arms]
    for index, seed in enumerate(seeds):
        order = labels[index % len(labels):]+labels[:index % len(labels)]
        names.extend(cells[label, seed] for label in order)
        reference = json.loads((ROOT/'configs'/(cells['zero', seed]+'.json')).read_text())
        reference['name'] = f'{tag}_zero_repeat_seed{seed}'
        save(reference)
        names.append(reference['name'])
    write(ROOT/'state'/(tag+'_confirmation_design.json'), dict(
        arms=dict(arms), arrival_seeds=seeds, jobs=names, frozen_epoch=frozen, output_cap=cap,
        scope='Held-out Arrival seeds. Prompt IDs and exact tokenized texts remain the frozen 128-prompt pool. Repeated Zero cells are independent process starts, not extra independent Arrival seeds.'))
    return names


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--arm', action='append', required=True,
        help='LABEL=COMPLETED_SELECTION_CONFIG. Put zero first.')
    parser.add_argument('--seeds', nargs='+', type=int, default=[2026110951, 2026110952, 2026110953])
    parser.add_argument('--tag', default='confirm')
    parser.add_argument('--cap', type=int, choices=(4, 16), default=16)
    args = parser.parse_args()
    names = prepare([tuple(value.split('=', 1)) for value in args.arm], args.seeds, args.tag, args.cap)
    print(' '.join(names))


if __name__ == '__main__':
    main()
