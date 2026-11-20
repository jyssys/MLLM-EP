"""Reproduce the client-visible tail symptom from preserved real traces."""
import argparse
import json
import numpy as np
from .common import ROOT, PREVIOUS, digest


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', default='Main_Native_LongBench_N200_q1_s2026112021_BI1_v1')
    parser.add_argument('--historical', action='store_true')
    parser.add_argument('--screen-only', action='store_true')
    args = parser.parse_args()
    folder = (PREVIOUS if args.historical else ROOT) / 'raw' / args.name
    records = [json.loads(line) for line in (folder/'requests.jsonl').read_text().splitlines()]
    intervals = [1000*(b['time']-a['time']) for row in records
                 for a,b in zip(row['token_events'],row['token_events'][1:])]
    window = json.loads((folder/'window.json').read_text())
    calls = [r for r in json.loads((folder/'NIXL_host_setup_decode.json').read_text())['rows']
             if r['kind']=='submit_transfer_host' and window['start']<=r['start']<=window['end']]
    values = [1000*(r['end']-r['start']) for r in calls]
    tail = float(np.percentile(intervals,99))
    symptom = tail > 34
    print(json.dumps(dict(name=args.name,historical=args.historical,requests=len(records),
        ITL_p99_ms=tail,ITL_threshold_ms=34,tail_symptom_reproduced=symptom,
        transfer_calls=len(calls),transfer_host_p95_ms=float(np.percentile(values,95)),
        requests_SHA256=digest(folder/'requests.jsonl'),
        scope='Replay of captured client delivery and API boundaries. API overlap alone is not a causal diagnosis.')))
    if not args.screen_only:
        assert not symptom, 'REPRODUCED: client-visible ITL p99 exceeds the frozen34ms threshold'


if __name__ == '__main__':
    main()
