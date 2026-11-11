"""Compare every delivered token in completed, matched serving cohorts."""
import argparse
from .common import *


def differences(reference, candidate):
    if len({row['seed'] for row in reference}) != len(reference):
        raise ValueError('Duplicate reference sampling seed')
    if len({row['seed'] for row in candidate}) != len(candidate):
        raise ValueError('Duplicate candidate sampling seed')
    a, b = ({row['seed']: row for row in rows} for rows in (reference, candidate))
    return [seed for seed in sorted(a.keys() | b.keys()) if seed not in a or seed not in b
        or a[seed]['input_ids'] != b[seed]['input_ids']
        or a[seed]['output_ids'] != b[seed]['output_ids']]


def compare(reference, candidate):
    initialize()
    before = json.loads((ROOT/'configs'/(reference+'.json')).read_text())
    after = json.loads((ROOT/'configs'/(candidate+'.json')).read_text())
    for field in ('model', 'model_revision', 'length', 'cap', 'depth', 'seed', 'count'):
        assert before[field] == after[field], f'Mismatched serving exactness field: {field}'
    def rows(name):
        path = ROOT/'raw'/name/'requests.jsonl'
        values = [json.loads(line) for line in path.read_text().splitlines()]
        values = [row for row in values if not row.get('warmup')]
        assert len(values) == before['count']
        return values
    failed = differences(rows(reference), rows(candidate))
    result = dict(reference=reference, candidate=candidate, expected=before['count'],
        divergent_sampling_seeds=failed, passed=not failed,
        scope='Every generated token and every input token in the completed serving cohort. This does not replace N8/N32 intermediate-state and lifetime qualification.')
    write(ROOT/'raw'/candidate/'serving_output_exactness.json', result)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('reference')
    parser.add_argument('candidates', nargs='+')
    args = parser.parse_args()
    print(json.dumps([compare(args.reference, name) for name in args.candidates]))


if __name__ == '__main__':
    main()
