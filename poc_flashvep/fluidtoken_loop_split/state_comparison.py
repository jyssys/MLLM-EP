"""Compare recorded valid prompt state without treating missing rows as equal."""
import argparse
from .common import *


def states(name, length):
    result = {}
    for path in (ROOT / 'raw' / name).glob('prompt_states_*.jsonl'):
        for line in path.read_text().splitlines():
            row = json.loads(line)
            if row['prompt_tokens'] != length:
                continue
            value = (row['hidden_sha256'], row['planes'])
            if row['seed'] in result:
                assert result[row['seed']] == value, 'Conflicting duplicate state'
            result[row['seed']] = value
    return result


def compare(reference, candidate):
    cfg = json.loads((ROOT / 'configs' / (candidate + '.json')).read_text())
    expected = set(range(cfg['seed'], cfg['seed'] + cfg['count']))
    left, right = states(reference, cfg['length']), states(candidate, cfg['length'])
    missing = sorted(expected - (left.keys() & right.keys()))
    different = sorted(seed for seed in expected & left.keys() & right.keys()
                       if left[seed] != right[seed])
    return dict(reference=reference, candidate=candidate, expected=len(expected),
                missing_seeds=missing, divergent_seeds=different,
                exact=bool(expected) and not missing and not different,
                scope='Exact hashes of recorded valid KV planes and prompt hidden state. Not original-HF accuracy.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('reference')
    parser.add_argument('candidate')
    args = parser.parse_args()
    result = compare(args.reference, args.candidate)
    write(ROOT / 'raw' / args.candidate / 'reference_prompt_state_comparison.json', result)
    print(json.dumps(result))
