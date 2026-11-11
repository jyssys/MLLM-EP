"""Summarize measured enclosures without calling gaps physical GPU idle."""
import statistics
from collections import defaultdict
from .common import *


def quantiles(values):
    values = sorted(values)
    if not values:
        return None
    from ..idle_intervention_dynamic_loop.analyze import percentile
    return dict(n=len(values), sum=sum(values), mean=statistics.mean(values),
        p50=percentile(values, .50), p95=percentile(values, .95),
        p99=percentile(values, .99), maximum=max(values))


def summarize(name):
    folder = ROOT / 'raw' / name
    cfg = json.loads((ROOT / 'configs' / (name+'.json')).read_text())
    expected = set(range(cfg['seed'], cfg['seed']+cfg['count']))
    roles = {}
    for path in folder.glob('sparse_device_*.json'):
        trace = json.loads(path.read_text())
        spans = defaultdict(list)
        prompt = []
        operation_costs = defaultdict(list)
        for row in trace['rows']:
            measured = [request for request in row['requests'] if request['seed'] in expected]
            if not measured:
                continue
            kind = 'prompt' if row['prompt'] else 'decode'
            key = f'{kind}/{row["operation"]}/rows{row["rows"]}/loops{row["loops"]}'
            operation_costs[key].append(row['CUDA_enclosure_ms'])
            if not row['prompt']:
                continue
            prompt.append(row['CUDA_enclosure_ms'])
            for request in measured:
                spans[request['seed']].append(row)
        summed, first_to_last, between = [], [], []
        for rows in spans.values():
            rows.sort(key=lambda r: r['begin']['time'])
            summed.append(sum(r['CUDA_enclosure_ms'] for r in rows))
            first_to_last.append(1000*(rows[-1]['end']['time']-rows[0]['begin']['time']))
            between.extend(1000*max(0., b['begin']['time']-a['end']['time']) for a, b in zip(rows, rows[1:]))
        roles[trace['role']] = dict(prompt_submission_count=len(prompt),
            prompt_submission_enclosure_ms=quantiles(prompt),
            per_request_sum_enclosure_ms=quantiles(summed),
            per_request_first_to_last_enclosure_ms=quantiles(first_to_last),
            per_request_between_enclosure_gap_ms=quantiles(between),
            measured_CUDA_enclosure_by_operation_ms={key:quantiles(vals)
                                                       for key,vals in sorted(operation_costs.items())},
            endpoint_clock_drift_ms=trace.get('endpoint_drift_ms'),
            event_record_host_s=trace.get('event_record_host_s'),
            decode_sample_period=trace.get('Decode_sample_period'),
            scope='Measured CUDA Event enclosures. Each member experiences the entire shared-batch enclosure. Per-request sums can therefore exceed the submission sum. Gaps can include other requests, host waiting, transport, and GPU activity. These are not kernel-union times or physical idle windows.')
    host = {json.loads(path.read_text())['role']: json.loads(path.read_text())['totals']
        for path in folder.glob('host_cost_*.json')}
    result = dict(name=name, roles=roles, inclusive_host=host,
        scope='Diagnostic cost control. Nested host counters overlap. Enclosures and host durations overlap. Do not sum them into latency.')
    write(ROOT / 'costs' / (name+'.json'), result)
    return result


def main():
    initialize()
    (ROOT / 'costs').mkdir(exist_ok=True)
    results = []
    for arm in ('zero', 'checkpoint', 'loop'):
        name = f'cost_{arm}_events_seed2026110931'
        result = ROOT / 'raw' / name / 'dispatcher_result.json'
        if result.exists() and json.loads(result.read_text()).get('complete'):
            results.append(summarize(name))
    write(ROOT / 'costs/controls.json', results)
    print(json.dumps([{k: v for k, v in row.items() if k in ('name', 'roles')} for row in results]))


if __name__ == '__main__':
    main()
