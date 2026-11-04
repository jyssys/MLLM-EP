"""Partition per-request host latency without adding overlapping GPU times."""
import argparse
import csv
import io
from .common import *

LABELS = ('admission_wait', 'P_dispatch_and_start_wait', 'P_observed_service',
          'P_completion_to_first_token', 'decode_generation')


def request_key(row):
    return f"r{row['run']}_q{row['sample_id']}"


def partition(row, metric):
    timings = metric['timings']
    origin = metric['created_monotonic']
    points = [row['arrival']] + [origin + timings[key] for key in
        ('admitted', 'prefill_started', 'prefill_compute_completed')]
    points += [row['first_token'], row['finish']]
    durations = [b - a for a, b in zip(points, points[1:])]
    if any(value < 0 for value in durations):
        raise ValueError('Host events are not a sequential partition')
    total = row['finish'] - row['arrival']
    if total <= 0 or abs(sum(durations) - total) > 1e-7:
        raise ValueError('Host partition does not conserve request E2E')
    return dict(zip(LABELS, durations)), total


def analyze(name):
    configure()
    folder = ROOT / 'raw' / name
    metrics = {r['request_id']: r for r in json.loads((folder / 'closed_stats.json').read_text())['metrics']}
    rows = [json.loads(line) for line in (folder / 'requests.jsonl').read_text().splitlines()]
    rows = [row for row in rows if not row.get('warmup')]
    output, failures = [], []
    sums = dict.fromkeys(LABELS, 0.)
    denominator = 0.
    for row in rows:
        request_id = request_key(row)
        try:
            durations, total = partition(row, metrics[request_id])
        except (KeyError, ValueError) as error:
            failures.append(dict(request_id=request_id, reason=str(error)))
            continue
        denominator += total
        for label, seconds in durations.items():
            sums[label] += seconds
            output.append(dict(request_id=request_id, component=label,
                               host_seconds=seconds, request_E2E_seconds=total,
                               request_E2E_percent=100 * seconds / total))
    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=('request_id', 'component', 'host_seconds',
                                               'request_E2E_seconds', 'request_E2E_percent'))
    writer.writeheader()
    writer.writerows(output)
    (folder / 'host_E2E_breakdown_v2.csv').write_text(buffer.getvalue())
    result = dict(name=name, measured_requests=len(rows), partitioned_requests=len(rows) - len(failures),
        failures=failures, evidence='MEASURED_HOST_EVENT_PARTITION', analysis_contract_version=2,
        summed_request_E2E_seconds=denominator,
        components={label: dict(host_seconds=value, aggregate_E2E_percent=100 * value / denominator if denominator else None)
                    for label, value in sums.items()},
        scope='Disjoint host intervals within each request. Shares use summed request latency, not server wall time or GPU critical-path time.',
        limitations=['Admission includes P/D resource and KV waiting.',
            'P observed service includes submission and scheduling gaps.',
            'P completion to first token includes exposed handoff, D continuation, scheduling, Coda, sampling, and output delivery.',
            'The activated receipt can follow the first output. It cannot divide the first-token critical path. Version 1 remains as a rejected chronology audit.',
            'Concurrent requests overlap. These sums are not exclusive server execution components.'])
    write(folder / 'host_E2E_breakdown_v2.json', result)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('name')
    args = parser.parse_args()
    configure()
    from ..looped_pd_autonomous_headroom.analysis_lane import quiet_analysis
    with quiet_analysis(args.name + '_host_breakdown'):
        print(json.dumps(analyze(args.name)))
