"""Identify rare output gaps that overlap a repeated prompt schedule."""
import collections
import json

from .analyze import records
from .common import ROOT, write


def analyze():
    results = []
    for folder in sorted((ROOT / 'raw').glob('confirm_DP2_B8192_S1024_*')):
        if not (folder / 'metrics.json').exists():
            continue
        cfg = json.loads((ROOT / 'configs' / (folder.name + '.json')).read_text())
        if cfg['arrival_rate'] < 7:
            continue
        spans, samples = collections.defaultdict(list), {}
        for path in folder.glob('native_*.json'):
            value = json.loads(path.read_text())
            for row in value.get('prompt_spans', []):
                spans[row['request_id'].split('-')[0]].append(row)
            samples[value.get('role')] = value.get('queue_samples', [])
        observed, total_intervals, total_ms = [], 0, 0.
        for request in records(folder / 'requests.jsonl'):
            for a, b in zip(request['token_events'], request['token_events'][1:]):
                delay = 1000 * (b['time'] - a['time'])
                total_intervals += 1
                total_ms += delay
                if delay < 1000:
                    continue
                rid = 'q' + str(request['sample_id'])
                repeated = [r for r in spans[rid] if a['worker_time'] <= r['time'] <= b['worker_time']]
                role = 'replica' + str(request['sample_id'] % 2)
                nearest = min(samples[role], key=lambda r: abs(r['time'] - a['worker_time']))
                first_repeat = min((r['time'] for r in repeated), default=None)
                observed.append(dict(request_id=rid, token_counts=[a['count'], b['count']],
                    parent_ITL_ms=delay, worker_ITL_ms=1000 * (b['worker_time'] - a['worker_time']),
                    delivery_lag_change_ms=delay - 1000 * (b['worker_time'] - a['worker_time']),
                    worker_begin=a['worker_time'], worker_end=b['worker_time'],
                    repeated_prompt_schedules=repeated,
                    before_repeat_schedule_ms=1000 * (first_repeat - a['worker_time']) if first_repeat else None,
                    repeat_schedule_to_next_worker_output_ms=1000 * (b['worker_time'] - first_repeat) if first_repeat else None,
                    nearest_queue_sample=nearest))
        results.append(dict(name=folder.name, QPS=cfg['arrival_rate'], seed=cfg['arrival_seed'],
            total_intervals=total_intervals, interval_fraction=len(observed) / total_intervals,
            observed_mean_TBT_component_ms=sum(v['parent_ITL_ms'] for v in observed) / total_intervals,
            fraction_of_summed_ITL_time=sum(v['parent_ITL_ms'] for v in observed) / total_ms,
            intervals_at_least_1s=len(observed), all_overlap_prompt_reschedule=bool(observed) and
                all(v['repeated_prompt_schedules'] for v in observed), outliers=observed))
    result = dict(starts=results, evidence='TRACE_ANALYSIS',
        scope='Actual worker/client output gaps and host prompt schedules. A completed prompt is scheduled again after decode already emitted tokens. This is consistent with KV-pressure preemption and recomputation. The exact preemption transition and exclusive GPU recompute time are NOT_MEASURED. The pre-reschedule period is not an exclusive GPU execution duration.')
    write(ROOT / 'analysis/fresh_primary_rare_ITL_prompt_reschedule.json', result)
    return result


if __name__ == '__main__':
    analyze()
