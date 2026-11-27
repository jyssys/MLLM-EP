"""Reduce native HTTP requests, exact cache counters and diagnostic pressure samples."""

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
from statistics import median

from .native_study import numeric_metrics


def read(path):
    return json.loads(path.read_text())


def rows(path):
    with path.open() as file:
        return [json.loads(line) for line in file]


def digest(path):
    with path.open('rb') as file:
        return hashlib.file_digest(file, 'sha256').hexdigest()


def pct(data, p):
    if not data:
        return None
    values = sorted(data)
    idx = (len(values) - 1) * p
    left = int(idx)
    return values[left] + (idx - left) * (values[min(left + 1, len(values) - 1)] - values[left])


def select_metric(values, *parts):
    return {key: value for key, value in values.items() if any(part in key for part in parts)}


def engine_deltas(before, after):
    return {role: {key: value - before[role].get(key, 0.0) for key, value in values.items()
                   if key.split(':', 2)[1].endswith('_total')} for role, values in after.items()}


def stage_intervals(proxy, label):
    picked = [row for row in proxy if row['runtime_request_id'].startswith(label + '_')]
    stages = {'encoder_HTTP': ('E_submit', 'E_complete'), 'prefill_HTTP': ('P_submit', 'P_ready'),
              'ready_to_D_submit': ('P_ready', 'D_submit'), 'D_submit_to_first_SSE': ('D_submit', 'D_first_SSE_bytes'),
              'decode_HTTP': ('D_submit', 'D_http_complete')}
    intervals = {key: [(row[right] - row[left]) * 1000 for row in picked if left in row and right in row]
                 for key, (left, right) in stages.items()}
    p_reported = []
    for row in picked:
        usage = row.get('P_response', {}).get('usage') or {}
        details = usage.get('prompt_tokens_details') or {}
        if details.get('cached_tokens') is not None:
            p_reported.append(details['cached_tokens'])
    return {
        'stages': {key: {'count': len(values), 'p50_ms': pct(values, .5), 'p95_ms': pct(values, .95),
                         'max_ms': max(values) if values else None} for key, values in intervals.items()},
        'P_usage_cache_entries': len(p_reported), 'P_usage_reported_cached_tokens': sum(p_reported),
        'P_usage_positive_cache_entries': sum(value > 0 for value in p_reported),
        'any_D_pending_at_P_ready': sum(bool((row.get('D_at_P_ready') or {}).get('pending_admissions', 0)) for row in picked),
        'scope': 'Host timeline. HTTP brackets can include queue, rendering, compute and transfer.',
    }


def p_submit_reuse(proxy, label, manifest):
    """Describe reuse order at the observed HTTP boundary, without engine assumptions."""
    if not manifest.is_file():
        return None
    groups = [tuple(image['SHA256'] for image in item['images']) for item in read(manifest)['requests']
              if all('SHA256' in image for image in item['images'])]
    entries = sorted((row for row in proxy if row['runtime_request_id'].startswith(label + '_')
                      and 'P_ready' in row and 'P_submit' in row), key=lambda row: row['P_submit'])
    if len(groups) != len(entries):
        return None
    seen, distances, ready_gaps = {}, [], []
    for order, row in enumerate(entries):
        index = int(row['runtime_request_id'].rsplit('_', 1)[-1])
        group = groups[index]
        if group in seen:
            previous, ready = seen[group]
            distances.append(order - previous)
            ready_gaps.append((row['P_submit'] - ready) * 1000)
        seen[group] = (order, row['P_ready'])
    return {'same_ordered_images_revisits': len(distances),
            'median_interarrival_distance_at_P_submit': pct(distances, .5),
            'prior_same_document_not_P_HTTP_ready_at_submit': sum(value < 0 for value in ready_gaps),
            'median_prior_same_document_P_ready_to_revisit_submit_ms': pct(ready_gaps, .5),
            'scope': 'Realized host order. Incomplete P HTTP does not imply that no block has been materialized.'}


def case(run, label, config, proxy):
    mfile = run / 'analysis' / f'{label}_metrics.json'
    m = read(mfile)
    source = run / 'raw' / f'{label}_prometheus.json'
    prom = read(source)
    deltas = engine_deltas(numeric_metrics(prom['before']), numeric_metrics(prom['after']))
    pressure_path = run / 'raw' / f'{label}_scheduler.jsonl'
    samples = [row['values'] for row in rows(pressure_path) if 'values' in row] if pressure_path.is_file() else []
    peaks = defaultdict(dict)
    for sample in samples:
        for role, values in sample.items():
            for key, value in select_metric(values, 'kv_cache_usage', 'num_requests_running', 'num_requests_waiting').items():
                peaks[role][key] = max(value, peaks[role].get(key, 0.0))

    encoder = {}
    if 'encoder_cache' in prom['after']:
        before, after = prom['before']['encoder_cache'][0], prom['after']['encoder_cache'][0]
        count_names = ('acquired_image_hits', 'acquired_image_misses', 'computed_image_saves', 'cache_evictions')
        encoder = {key: after[key] - before[key] for key in count_names}
        encoder.update({key: after[key] for key in ('idle_storage_bytes', 'active_request_leases', 'retained_storage_bytes')})
        for key, value in after.items():
            if (key.startswith(('host_store_', 'host_load_'))
                    or key in ('host_loads', 'host_stores', 'host_evictions')) and isinstance(value, (int, float)):
                encoder[key] = value - before.get(key, 0)
            elif key.startswith('host_') or key == 'retained_cache_device':
                encoder[key] = value
    times = {key: m.get(key) for key in ('TTFT_p50_ms', 'TTFT_p95_ms', 'TTFT_p99_ms', 'TPOT_p95_ms', 'ITL_p99_ms')}
    data = sorted(rows(run / 'raw' / f'{label}_requests.jsonl'), key=lambda row: row['index'])
    all_tpots = [row['TPOT_ms'] for row in data if row['status'] == 'MEASURED_ONLINE']
    all_itls = [gap for row in data if row['status'] == 'MEASURED_ONLINE' for gap in row['ITL_ms']]
    e2e = [row['E2E_ms'] for row in data if row['status'] == 'MEASURED_ONLINE']
    for key, values in [('E2E', e2e), ('TPOT', all_tpots), ('ITL', all_itls)]:
        for quantile in ('50', '95', '99'):
            times[f'{key}_p{quantile}_ms'] = pct(values, int(quantile) / 100)
    successes = m['categories'].get('JOINT_PASS', 0)
    arrival = run / 'configs' / f'{label}_arrival.json'
    host = stage_intervals(proxy, label)
    cases = [('wave1', data[:94]), ('wave2', data[94:])] if len(data) == 188 else []
    wave_comparison = {key: {'TTFT_p95_ms': pct([row['TTFT_ms'] for row in section if 'TTFT_ms' in row], .95),
                             'n': len(section)} for key, section in cases}
    return {
        'run': run.name, 'cohort': label, 'evidence': 'MEASURED_ONLINE' if m['primary_execution_valid'] else 'FAILED_EXECUTION',
        'manifest_sha256': config['manifest_SHA256'], 'arrival_sha256': digest(arrival), 'raw_sha256': m['raw_SHA256'],
        'params': {key: config['args'].get(key) for key in ('tier', 'tier_gib', 'p_kv_gib', 'd_kv_gib', 'stable', 'encoder_cache',
                   'apc_roles', 'qps', 'output', 'cache_warm_count', 'encoder_cache_device', 'ec_gib', 'identity')},
        'scheduled': m['planned_count'], 'completed': m['completed_count'], 'failed': m['failed_count'],
        'finite_cohort_time_s': m['duration_seconds'], 'joint_slo_requests': successes,
        'joint_goodput_req_s': successes / m['duration_seconds'], 'joint_output_goodput_tok_s': m['SLO_goodput'],
        'completion_req_s': m['completed_requests_per_second'], 'output_tok_s': m['raw_output_tokens_per_second'],
        'latency': times, 'per_wave': wave_comparison, 'cache_counters': encoder,
        'engine_counters': {role: select_metric(values, 'prefix_cache', 'num_preemptions', 'kv_offload', 'nixl', 'queue')
                            for role, values in deltas.items()},
        'P_actual_submission_order': p_submit_reuse(proxy, label, run / 'data/frozen_manifest.json'),
        'pressure_poll_samples': len(samples), 'pressure_sample_peak': peaks, 'host_stage': host,
        'request_ids_sha256': hashlib.sha256('\n'.join(row['request_id'] for row in data).encode()).hexdigest(),
    }


def scan(run):
    if not (run / 'state/environment.json').is_file():
        return {'run': run.name, 'status': 'NOT_RUN'}
    config = read(run / 'state/environment.json')
    failure_path = run / 'state/failure.json'
    proxy_file = run / 'raw/proxy_epd.jsonl'
    proxy = rows(proxy_file) if proxy_file.is_file() else []
    labels = [path.name.removesuffix('_metrics.json') for path in sorted((run / 'analysis').glob('measured_*_metrics.json'))]
    worker_file = run / 'state/workers_after_drain.json'
    memory_fields = ('allocated_bytes', 'reserved_bytes', 'maximum_allocated_bytes', 'device_free_bytes',
                     'device_total_bytes', 'unique_KV_storage_bytes', 'model_parameter_logical_bytes')
    stage_memory = ({role: {key: worker[key] for key in memory_fields} for role, worker in read(worker_file).items()}
                    if worker_file.is_file() else None)
    peer_file = run / 'state/peer_storage.json'
    return {'run': run.name, 'status': 'COMPLETE' if (run / 'state/complete.json').is_file()
            else 'FAILED' if failure_path.is_file() else 'INCOMPLETE',
            'failure': read(failure_path) if failure_path.is_file() else None,
            'stage_memory_after_drain': stage_memory,
            'physical_peer_storage': read(peer_file) if peer_file.is_file() else None,
            'measurements': [case(run, label, config, proxy) for label in labels]}


def fidelity(reference, candidate, tag):
    first, separator, second = tag.partition(':')
    second = second if separator else first
    baseline = {row['request_id']: row for row in rows(reference / 'raw' / f'{first}_requests.jsonl')}
    chosen = {row['request_id']: row for row in rows(candidate / 'raw' / f'{second}_requests.jsonl')}
    if baseline.keys() != chosen.keys():
        raise ValueError('Correctness pairs must use the same frozen request IDs')
    both = [key for key in baseline if baseline[key]['status'] == chosen[key]['status'] == 'MEASURED_ONLINE']
    exact = [key for key in both if baseline[key]['token_ids'] == chosen[key]['token_ids']]
    first = [key for key in both if baseline[key]['token_ids'][0] == chosen[key]['token_ids'][0]]
    return {'reference': reference.name, 'candidate': candidate.name, 'tag': tag, 'n': len(baseline),
            'paired_successes': len(both), 'exact_generation': len(exact), 'exact_first_token': len(first),
            'nonexact_request_ids': [key for key in both if key not in exact],
            'scope': 'Client tokens in Native BI0 with original continuous batching. Exact output fidelity alone is not task quality.'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, action='append', required=True)
    parser.add_argument('--fidelity', nargs=3, metavar=('REFERENCE', 'CANDIDATE', 'TAG'), action='append', default=[])
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    result = {'runs': [scan(run) for run in args.run],
              'fidelity': [fidelity(Path(left), Path(right), tag) for left, right, tag in args.fidelity]}
    args.out.write_text(json.dumps(result, indent=2) + '\n')


if __name__ == '__main__':
    main()
