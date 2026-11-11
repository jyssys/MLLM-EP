"""Count actual queued descriptor payloads without treating them as DMA time."""
import argparse
from .common import *
from .cost_analysis import quantiles


def summarize(name):
    cfg = json.loads((ROOT/'configs'/(name+'.json')).read_text())
    folder = ROOT/'raw'/name
    begin = json.loads((folder/'measurement_begin.json').read_text())['time']
    expected = set(range(cfg['seed'], cfg['seed']+cfg['count']))
    rows = [row for path in folder.glob('migration_*.jsonl')
        for line in path.read_text().splitlines() for row in [json.loads(line)]
        if row['time'] >= begin and row['seed'] in expected]
    # Startup requests can reuse a measured sampling seed. Both filters are
    # required. More than one publication per seed must remain explicit.
    result = dict(name=name, measured_requests=cfg['count'], publications=len(rows),
        unique_sampling_seeds=len({row['seed'] for row in rows}),
        KV_bytes=sum(row['KV_bytes'] for row in rows),
        hidden_bytes=sum(row['hidden_bytes'] for row in rows),
        total_descriptor_payload_bytes=sum(row['KV_bytes']+row['hidden_bytes'] for row in rows),
        per_publication_total_bytes=quantiles([row['KV_bytes']+row['hidden_bytes'] for row in rows]),
        per_publication_descriptors=quantiles([row['descriptors'] for row in rows]),
        rows=rows,
        scope='Observed queued exact-state descriptor payloads. These are not measured DMA elapsed times, hardware link counters, or additional cost relative to Native. Native transfer must be subtracted explicitly. Immutable early publication requires its separate delta-publication ledger.')
    write(ROOT/'transfer_accounting'/(name+'.json'), result)
    return {key:value for key,value in result.items() if key != 'rows'}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('names', nargs='+')
    args = parser.parse_args()
    initialize()
    print(json.dumps([summarize(name) for name in args.names]))


if __name__ == '__main__':
    main()
