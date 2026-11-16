"""Distinguish unique prompt coverage from scheduled recomputation under KV pressure."""
import collections
import json

from .common import ROOT, write
from .kernel_trace import duration, union


def analyze(name):
    config = json.loads((ROOT / 'configs' / (name + '.json')).read_text())
    folder = ROOT / 'raw' / name
    requests = collections.defaultdict(list)
    included_roles = set()
    for path in folder.glob('native_*.json'):
        value = json.loads(path.read_text())
        if 'prompt_spans' not in value:
            continue
        if config['arm'] == 'PD_G1' and value.get('role') != 'prefill':
            continue  # D's source-token priming is not another model Prefill.
        included_roles.add(value.get('role', path.stem))
        for row in value.get('prompt_spans', []):
            rid = row['request_id'].split('-')[0]
            if not (rid.startswith('q') and rid[1:].isdigit()):
                continue
            requests[rid].append((row['start'], row['start'] + row['count']))
    per_request = {}
    for rid, intervals in sorted(requests.items()):
        unique = duration(union(intervals, 0, config['length']))
        submitted = sum(end - start for start, end in intervals)
        per_request[rid] = dict(unique_prompt_rows=unique, scheduled_prompt_rows=submitted,
            duplicate_scheduled_prompt_rows=submitted - unique, scheduled_spans=len(intervals))
    expected = config['count'] * config['length']
    unique = sum(row['unique_prompt_rows'] for row in per_request.values())
    scheduled = sum(row['scheduled_prompt_rows'] for row in per_request.values())
    result = dict(name=name, included_roles=sorted(included_roles), unique_prompt_rows=unique, expected_prompt_rows=expected,
        scheduled_prompt_rows=scheduled, duplicate_scheduled_prompt_rows=scheduled - unique,
        duplicate_fraction_of_expected=(scheduled - unique) / expected,
        complete_unique_prompt_coverage=unique == expected,
        repeated_prompt_requests=[rid for rid, row in per_request.items() if row['duplicate_scheduled_prompt_rows']],
        requests=per_request, evidence='TRACE_ANALYSIS',
        scope='Host scheduler spans, not kernel FLOPs. Duplicate scheduled rows can indicate KV-pressure recomputation. An actual model-input trace is needed to establish execution of each repeated row. This does not count ordinary distinct token chunks as duplicate work.')
    write(ROOT / 'analysis' / (name + '_scheduled_work.json'), result)
    return result


if __name__ == '__main__':
    for path in sorted((ROOT / 'raw').glob('*/metrics.json')):
        value = json.loads(path.read_text())
        if value.get('complete') and value.get('output_gate', {}).get('passed'):
            analyze(path.parent.name)
