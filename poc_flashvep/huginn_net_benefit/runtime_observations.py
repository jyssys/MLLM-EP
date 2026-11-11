"""Separate observed queue, transfer, device samples, and policy counters."""
import argparse
import statistics
from collections import Counter
from .common import *
from .cost_analysis import quantiles


def records(path):
    if path.exists():
        with path.open() as handle:
            yield from map(json.loads, handle)


def fraction(values):
    return statistics.mean(values) if values else None


def summarize(name):
    folder = ROOT/'raw'/name
    cfg_path = ROOT/'configs'/(name+'.json')
    cfg = json.loads(cfg_path.read_text())
    window = json.loads((folder/'window.json').read_text())
    start, last, stop = (window[key] for key in ('start', 'last_arrival', 'end'))
    left, right = start+.2*(last-start), start+.9*(last-start)
    admission = [r for r in records(folder/'admission_reasons.jsonl') if left <= r['time'] <= right]
    p_only, d_only, waiting, reason_counts = [], [], [], Counter()
    for sample in admission:
        blocked = [set(row['reasons']) for row in sample['waiting']]
        waiting.append(len(blocked))
        p_only.append(any(any(k.startswith('PREFILL_NO_') for k in row)
            and not any(k.startswith('DECODE_NO_') for k in row) for row in blocked))
        d_only.append(any('DECODE_NO_KV_CREDIT' in row
            and not any(k.startswith('PREFILL_NO_') for k in row) for row in blocked))
        reason_counts.update(key for row in blocked for key in row)
    credits = [r for r in records(folder/'queue_observations.jsonl') if left <= r['time'] <= right]
    role_samples = {}
    for role in ('prefill', 'decode'):
        entries = [v for r in credits for v in r['worker_credits'].values() if v['role'] == role]
        role_samples[role] = dict(reserved_blocks=quantiles([r['reserved_blocks'] for r in entries]),
            reserved_slots=quantiles([r['slots'] for r in entries]),
            total_blocks=max((r['total_blocks'] for r in entries), default=None))

    device = {}
    for role, gpu in zip(('prefill', 'decode'), cfg['gpus']):
        samples = [r for path in folder.glob('NVML-*.jsonl') for r in records(path)
            if r['physical_GPU'] == gpu and start <= r['time'] <= stop]
        util = [r['sampled_utilization_percent'] for r in samples if left <= r['time'] <= right]
        device[role] = dict(physical_GPU=gpu, central_NVML_utilization_samples=quantiles(util),
            measured_cohort_peak_HBM_bytes=max((r['used_HBM_bytes'] for r in samples), default=None),
            physical_kernel_busy_percent='NOT_MEASURED', physical_idle_windows='NOT_MEASURED')

    decode_batches = Counter()
    measured_D_prompt_seeds = set()
    for path in folder.glob('atomic_spans_*.json'):
        value = json.loads(path.read_text())
        if value['role'] != 'decode':
            continue
        measured_D_prompt_seeds.update(row['seed'] for row in value['prompt_spans']
            if start <= row['time'] <= stop and 0 <= row['seed']-cfg['seed'] < cfg['count']
            and row['loop_count'] > 0)
        for row in value['time_histogram']:
            if (left <= row['time'] and row['time']+.5 <= right
                    and row['operation'] == 'recurrent' and row['logical_rows'] == row['requests']):
                decode_batches[row['requests']] += row['submissions']

    expected = {f'r0_q{row["sample_id"]}' for row in records(folder/'requests.jsonl') if not row.get('warmup')}
    native_rows = [row for row in json.loads((folder/'stats.json').read_text())['metrics']
                   if row['request_id'] in expected]
    timing = [row['timings'] for row in native_rows]
    active_continuations = []
    for row in native_rows:
        index = int(row['request_id'].split('_q')[1])
        times = row['timings']
        if cfg['seed']+index in measured_D_prompt_seeds and 'd_received' in times and 'first_token' in times:
            origin = row['created_monotonic']
            active_continuations.append((origin+times['d_received'],origin+times['first_token']))
    active_continuations.sort()
    pairs = dict(coordinator_admission_wait=(None, 'admitted'),
        source_start_receipt=(None, 'prefill_started'),
        P_source_start_to_compute_release=('prefill_started', 'prefill_compute_completed'),
        P_compute_release_to_wire_commit=('prefill_compute_completed', 'prefill_submitted'),
        wire_commit_to_D_received=('prefill_submitted', 'd_received'),
        D_received_to_activated=('d_received', 'd_activated'),
        D_received_to_first_output=('d_received', 'first_token'))
    components = {}
    for label, (a,b) in pairs.items():
        vals = [1000*(row[b]-row.get(a, 0.)) for row in timing if b in row and (a is None or a in row)]
        components[label] = dict(ms=quantiles(vals), negative_order_observations=sum(v < 0 for v in vals))

    pipeline = [r for p in folder.glob('source_pipeline_*.json')
        for r in json.loads(p.read_text())['records']]
    forecasts = [r for p in folder.glob('predictive_checkpoint_preparation_*.json')
        for r in json.loads(p.read_text())['records'] if start <= r['time'] <= stop]
    result = dict(name=name, configuration_sha256=digest(cfg_path),
        time_scope=dict(central_arrival_window=[left,right], measurement=[start,stop],
            sample_scope='The central window covers 20% through 90% of the arrival span. These are sample fractions, not continuous time fractions.'),
        admission=dict(samples=len(admission), waiting_requests=quantiles(waiting),
            P_only_blocked_sample_fraction=fraction(p_only), D_KV_only_blocked_sample_fraction=fraction(d_only),
            reason_sample_occurrences=dict(reason_counts),
            scope='P-only means a Prefill slot or KV reason without a Decode reason. It is not measured GPU saturation.'),
        coordinator_credits=role_samples, sampled_GPU=device,
        observed_D_decode_batch_histogram=dict(decode_batches),
        D_continuation_host_cohort=dict(
            observed=len(active_continuations),
            D_receive_interarrival_ms=quantiles([1000*(b[0]-a[0])
                for a,b in zip(active_continuations,active_continuations[1:])]),
            D_receive_to_output_ms=quantiles([1000*(b-a) for a,b in active_continuations]),
            intersecting_receive_to_output_intervals=sum(
                any(i != j and a < d and c < b for j,(c,d) in enumerate(active_continuations))
                for i,(a,b) in enumerate(active_continuations)),
            scope='Host receipt enclosures for requests with actual D recurrent prompt spans. They do not measure physical ready windows.'),
        host_components=components, host_measured_requests=len(timing),
        source_pipeline=dict(progress_calls=len(pipeline), inclusive_host_ms=quantiles([r['host_elapsed_ms'] for r in pipeline]),
            safe_messages_on_recorded_progress_calls=sum(r['control_messages'] for r in pipeline)),
        predictive_preparation_calls=len(forecasts),
        scope='Native Host receipts and NVML samples. Prefill-start is a Host receipt. Wire commit means all fragments were submitted, not all DMA completed. Intervals can overlap. Do not sum their percentiles. No exact Service TTFT or physical kernel idle follows from these values.')
    write(ROOT/'runtime_observations'/(name+'.json'), result)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('names', nargs='+')
    args = parser.parse_args()
    initialize()
    print(json.dumps([summarize(name) for name in args.names]))


if __name__ == '__main__':
    main()
