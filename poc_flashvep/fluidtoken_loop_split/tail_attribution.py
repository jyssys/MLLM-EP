"""Report temporal overlap, without treating overlap as proof of causation."""
import argparse
from .common import *


def analyze(name):
    configure()
    folder = ROOT / 'raw' / name
    capture = json.loads((folder / 'activity_capture.json').read_text())
    left = capture['start']['monotonic_s']
    right = capture['stop']['monotonic_s']
    probe_file = folder / 'D_probe_requests.jsonl'
    probes = [json.loads(x) for x in probe_file.read_text().splitlines()] if probe_file.exists() else []
    threshold = json.loads((ROOT / 'configs/frozen_SLO.json').read_text())['huginn_L2048']['ITL_ms']
    events = []
    for line in (folder / 'requests.jsonl').read_text().splitlines():
        row = json.loads(line)
        if row.get('warmup'):
            continue
        for event in row['token_events']:
            if event['first']:
                continue
            a, b = event['previous_time'], event['time']
            if not left <= a <= b <= right:
                continue
            value = 1000 * (b - a) / event['new_tokens']
            events.append(dict(sample_id=row['sample_id'], start=a, end=b, ITL_ms=value,
                violation=value > threshold,
                overlapping_probe_ids=[p['request_id'] for p in probes if p['time'] < b and p['finish'] > a]))
    result = dict(name=name, window_start=left, window_end=right,
        ITL_limit_ms=threshold, events=len(events),
        violations=sum(e['violation'] for e in events),
        violations_with_probe_overlap=sum(e['violation'] and bool(e['overlapping_probe_ids']) for e in events),
        largest_intervals=sorted(events, key=lambda x: -x['ITL_ms'])[:20],
        scope='Host token intervals and probe lifetimes from the same run. Temporal overlap is not causal attribution.')
    write(folder / 'tail_probe_overlap.json', result)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('name')
    args = parser.parse_args()
    configure()
    from ..looped_pd_autonomous_headroom.analysis_lane import quiet_analysis
    with quiet_analysis(args.name + '_tail_overlap'):
        print(json.dumps(analyze(args.name)))
