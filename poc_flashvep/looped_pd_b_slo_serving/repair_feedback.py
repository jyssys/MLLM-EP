"""Exact natural-output regression at the actual PD engine seam."""
import argparse
import json

from .common import ROOT, write
from .jobs import parity


def check(left, right, expected):
    result = parity(left, right)
    result['passed'] = result['N'] == expected and result['exact'] == expected and result['complete_ids']
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--left', required=True)
    parser.add_argument('--right', required=True)
    parser.add_argument('--expected', required=True, type=int)
    parser.add_argument('--result-name')
    args = parser.parse_args()
    result = check(args.left, args.right, args.expected)
    if args.result_name:
        write(ROOT / 'configs' / (args.result_name + '.json'), result)
    print(json.dumps(result, indent=2))
    if not result['passed']:
        raise SystemExit('EXACT_GENERATION_REGRESSION_FAILED')


if __name__ == '__main__':
    main()
