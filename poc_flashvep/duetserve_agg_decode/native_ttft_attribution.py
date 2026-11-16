"""Measure the Native cohort-ingress TTFT path as a host-time identity."""
import collections
import json
import statistics

from .analyze import records
from .common import ROOT, write
from .statistics import mean_ci


def components(arrival, source_ready, destination_send_end, first_token):
    return dict(arrival_to_source_ready_ms=1000 * (source_ready - arrival),
        source_ready_to_D_socket_end_ms=1000 * (destination_send_end - source_ready),
        D_socket_end_to_first_output_ms=1000 * (first_token - destination_send_end))


def analyze():
    starts, grouped = [], collections.defaultdict(list)
    for path in sorted((ROOT / 'raw').glob('confirm_Native_*/metrics.json')):
        metric = json.loads(path.read_text())
        assert metric['complete'] and metric['output_gate']['passed']
        folder = path.parent
        cfg = json.loads((ROOT / 'configs' / (folder.name + '.json')).read_text())
        handoffs = {r['request_id']: r['time'] for r in records(folder / 'native_handoffs.jsonl')}
        sends = {r['request_id']: r['actual_send_end'] for r in records(folder / 'D_actual_submissions.jsonl')}
        values = []
        for row in records(folder / 'requests.jsonl'):
            rid = 'q' + str(row['sample_id'])
            value = components(row['arrival'], handoffs[rid], sends[rid], row['first_token'])
            assert abs(sum(value.values()) - row['ttft_ms']) < 1e-6
            values.append(value)
        summary = {key: statistics.mean(r[key] for r in values) for key in values[0]}
        summary.update(name=folder.name, seed=cfg['arrival_seed'], QPS=cfg['arrival_rate'],
            requests=len(values), mean_TTFT_ms=metric['mean_TTFT_ms'],
            D_host_wait_fraction_of_TTFT=summary['source_ready_to_D_socket_end_ms'] / metric['mean_TTFT_ms'])
        starts.append(summary)
        grouped[cfg['arrival_rate']].append(summary)
    keys = ('arrival_to_source_ready_ms', 'source_ready_to_D_socket_end_ms',
            'D_socket_end_to_first_output_ms', 'mean_TTFT_ms', 'D_host_wait_fraction_of_TTFT')
    result = dict(starts=starts, QPS_summary=[dict(QPS=q, metrics={k: mean_ci(v[k] for v in rows) for k in keys})
        for q, rows in sorted(grouped.items())], evidence='TRACE_ANALYSIS',
        scope='Exact host-time TTFT decomposition. Source-ready includes P queue, compute, and notification. Destination waiting includes the cohort-ingress guard and IPC. The final segment includes import, barrier, and output delivery. These are not exclusive GPU costs or wire-transfer durations. The Native baseline is cohort-isolated, not a standard continuous-D-ingress frontier.')
    write(ROOT / 'analysis/native_cohort_ingress_TTFT_attribution.json', result)
    return result


if __name__ == '__main__':
    analyze()
