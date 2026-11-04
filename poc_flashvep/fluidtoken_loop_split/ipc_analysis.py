"""Associate slow IPC calls with retained token intervals in one run."""
import argparse
from collections import defaultdict
from .common import *


def analyze(name):
    folder = ROOT / 'raw' / name
    window = json.loads((folder / 'window.json').read_text())
    events = []
    groups = defaultdict(list)
    for path in folder.glob('host_pauses_*.json'):
        value = json.loads(path.read_text())
        for event in value['events']:
            if window['start'] <= event['start'] <= window['end']:
                row = dict(event, pid=value['pid'])
                events.append(row)
                groups[(value['pid'], event['method'], event.get('message_kind'))].append(row)
    result = dict(name=name, evidence='MEASURED_HOST_INTERVALS',
        scope='Nested calls overlap. Their durations are not additive GPU or E2E components.',
        causal_status='Temporal association alone does not prove a cause.',
        groups=[dict(pid=key[0], method=key[1], message_kind=key[2], count=len(rows),
                     max_seconds=max(row['seconds'] for row in rows),
                     summed_enclosures_seconds=sum(row['seconds'] for row in rows))
                for key, rows in groups.items()],
        longest=sorted(events, key=lambda row: row['seconds'], reverse=True)[:100])
    write(folder / 'ipc_pause_analysis.json', result)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('name')
    args = parser.parse_args()
    configure()
    result = analyze(args.name)
    print(json.dumps(result['groups'], indent=2))
