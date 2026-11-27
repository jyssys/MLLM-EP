"""Compute paired deltas with the independent server start as the resampling unit."""

import argparse
import json
from pathlib import Path
import random
from statistics import mean

from .online_summary import pct, scan


METRICS = ('TTFT_p95_ms', 'TPOT_p95_ms', 'ITL_p99_ms', 'E2E_p95_ms', 'joint_goodput_req_s')


def compare(blocks, cohort, resamples=10_000):
    if len({path.resolve() for block in blocks for path in block}) != 2 * len(blocks):
        raise ValueError('Each paired comparison must use distinct server starts')
    pairs = []
    for baseline, peer in blocks:
        reference, candidate = (next(row for row in scan(path)['measurements'] if row['cohort'] == cohort)
                                for path in (baseline, peer))
        for field in ('manifest_sha256', 'arrival_sha256', 'request_ids_sha256'):
            if reference[field] != candidate[field]:
                raise ValueError(f'Paired inputs differ in {field}')
        for field in ('qps', 'output', 'd_kv_gib'):
            if reference['params'][field] != candidate['params'][field]:
                raise ValueError(f'Paired workload contract differs in {field}')
        if reference['failed'] or candidate['failed']:
            raise ValueError('Performance ratios require complete matching cohorts; retain failures in the main ledger')
        pairs.append((reference, candidate))
    ratios = {}
    for key in METRICS:
        observed = []
        for reference, candidate in pairs:
            a = reference['latency'][key] if key in reference['latency'] else reference[key]
            b = candidate['latency'][key] if key in candidate['latency'] else candidate[key]
            observed.append(b / a if a > 0 else None)
        finite = [value for value in observed if value is not None]
        bootstrap = []
        if len(finite) == len(pairs) >= 3:
            rng = random.Random(2026112805)
            bootstrap = [mean(rng.choices(finite, k=len(finite))) for _ in range(resamples)]
        ratios[key] = {'ratio_by_start': observed, 'mean_ratio': mean(finite) if finite else None,
                       'paired_start_bootstrap_CI95': [pct(bootstrap, .025), pct(bootstrap, .975)] if bootstrap else None}
    return {'cohort': cohort, 'independent_pairs': len(pairs),
            'pairs': [[left['run'], right['run']] for left, right in pairs], 'ratios': ratios,
            'scope': 'Resamples server-start blocks. No independent-start CI below three complete pairs. '
                     'The inference applies to the frozen request trace and offered load.'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--pair', type=Path, nargs=2, action='append', required=True)
    parser.add_argument('--cohort', default='measured_1')
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    args.out.write_text(json.dumps(compare(args.pair, args.cohort), indent=2) + '\n')


if __name__ == '__main__':
    main()
