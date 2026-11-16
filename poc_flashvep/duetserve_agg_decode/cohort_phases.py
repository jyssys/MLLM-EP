"""Measure finite-cohort timing composition without calling it GPU idle."""
import json
import statistics

from .analyze import records
from .common import ROOT, write


def analyze(name):
    folder = ROOT / 'raw' / name
    config = json.loads((ROOT / 'configs' / (name + '.json')).read_text())
    rows = records(folder / 'requests.jsonl')
    window = json.loads((folder / 'window.json').read_text())
    arrival_end = window['last_arrival']
    if config['arm'] in ('DP2', 'TP2'):
        prompt_end = max(row['token_events'][0]['worker_time'] for row in rows)
        boundary_scope = 'Last local full-Prefill first-token emission. Not a CUDA boundary.'
    else:
        prompt_end = max(row['time'] for row in records(folder / 'native_handoffs.jsonl'))
        boundary_scope = 'Last P Source-ready notification. D import can still remain pending.'
    groups = {key: [] for key in ('ACTIVE_ARRIVALS', 'POST_ARRIVAL_PROMPTS_PENDING', 'NO_NEW_PROMPT_TAIL', 'BOUNDARY_CROSSING')}
    for row in rows:
        for a, b in zip(row['token_events'], row['token_events'][1:]):
            begin, end = a['worker_time'], b['worker_time']
            value = 1000 * (b['time'] - a['time'])
            if end <= arrival_end:
                key = 'ACTIVE_ARRIVALS'
            elif begin >= max(arrival_end, prompt_end):
                key = 'NO_NEW_PROMPT_TAIL'
            elif begin >= arrival_end and end <= prompt_end:
                key = 'POST_ARRIVAL_PROMPTS_PENDING'
            else:
                key = 'BOUNDARY_CROSSING'
            groups[key].append(value)
    total = sum(len(values) for values in groups.values())
    result = dict(name=name, last_arrival=arrival_end, last_prompt_notification=prompt_end,
        groups={key: dict(intervals=len(values), interval_fraction=len(values) / total,
            mean_TBT_ms=statistics.mean(values) if values else None) for key, values in groups.items()},
        boundary_scope=boundary_scope, evidence='TRACE_ANALYSIS',
        scope='Observed client/worker boundaries only. A post-arrival drain can still contain Prefill. A no-new-prompt tail can contain KV work or recomputation. No GPU-idle claim.')
    write(ROOT / 'analysis' / (name + '_cohort_phases.json'), result)
    return result


if __name__ == '__main__':
    for folder in sorted((ROOT / 'raw').iterdir()):
        if folder.name.startswith(('sweep_', 'concurrency_', 'confirm_', 'long_')) and (folder / 'metrics.json').exists():
            analyze(folder.name)
