"""Compare wide-campaign CPC traces and preserve the original motivation gates."""

import argparse
import itertools
import json
from pathlib import Path
import statistics

from ..mllm_epd_h200_prefill_heavy_batch_matched.capacity import percentile
from ..mllm_epd_h200_prefill_heavy_batch_matched.discovery import evaluate as discovery_gate
from ..mllm_epd_h200_prefill_heavy_batch_matched.trace import (
    COMPARISON_METRICS, complement, elapsed, intersect, merged, scheduler_intervals,
)


WINDOW_MS = (32, 64, 100, 128, 160, 225, 300, 500)
CPC_PHASES = ('clean_before', 'profile', 'clean_after')


def read(path):
    return json.loads(path.read_text())


def jsonl(path):
    with path.open() as stream:
        return [json.loads(line) for line in stream if line.strip()]


def paired_rows(run, tag):
    rows = jsonl(run / 'raw' / f'{tag}_requests.jsonl')
    ids = {row['runtime_request_id'] for row in rows}
    proxy = [row for row in jsonl(run / 'raw/proxy_epd.jsonl') if row['runtime_request_id'] in ids]
    if len(proxy) != len({row['runtime_request_id'] for row in proxy}):
        raise ValueError(f'Duplicate proxy request in {run.name}/{tag}')
    return rows, proxy


def distribution(values):
    if not values:
        return None
    return {'n': len(values), 'mean': statistics.mean(values), 'p50': percentile(values, .5),
            'p95': percentile(values, .95), 'p99': percentile(values, .99), 'max': max(values)}


def coverage(windows, p_busy, p_busy_waiting, total, threshold):
    eligible = [(a, b) for a, b in windows if b - a >= threshold / 1000]
    seconds = elapsed(eligible)
    p_seconds = elapsed(p_busy)
    overlap = elapsed(intersect(p_busy, eligible))
    waiting_overlap = elapsed(intersect(p_busy_waiting, eligible))
    return {'windows': len(eligible), 'window_seconds': seconds, 'window_fraction_all': seconds / total,
            'P_busy_overlap_seconds': overlap, 'P_busy_overlap_fraction_all': overlap / total,
            'P_busy_overlap_fraction_P_busy': overlap / p_seconds if p_seconds else None,
            'P_busy_waiting_overlap_seconds': waiting_overlap,
            'P_busy_waiting_overlap_fraction_all': waiting_overlap / total,
            'P_busy_waiting_overlap_fraction_P_busy': waiting_overlap / p_seconds if p_seconds else None}


def decode_slo(rows, planned):
    if planned <= 0 or len(rows) > planned:
        raise ValueError('Invalid planned Decode SLO denominator')
    completed = [row for row in rows if row['status'] == 'MEASURED_ONLINE']
    itl = [all(gap <= 34 for gap in row['ITL_ms']) for row in completed]
    tpot = [row['TPOT_ms'] <= 33 for row in completed]
    decoded = sum(a and b for a, b in zip(itl, tpot, strict=True))
    return {'planned': planned, 'completed': len(completed), 'ITL_pass': sum(itl), 'TPOT_pass': sum(tpot),
            'both_pass': decoded, 'attainment': decoded / planned, 'meets_95_percent': decoded / planned >= .95,
            'thresholds_ms': {'per_client_ITL': 34, 'TPOT': 33}}


def runtime_stages(clients, proxy):
    """Summarize observed host boundaries, distinguishing candidate from Native rows."""
    by_id = {row['runtime_request_id']: row for row in clients if row['status'] == 'MEASURED_ONLINE'}
    groups = {'all': [], 'actual_split': [], 'native_or_fallback': []}
    for row in proxy:
        client = by_id.get(row['runtime_request_id'])
        if client is not None and row['status'] == 'MEASURED_ONLINE':
            groups['all'].append((client, row))
            groups['actual_split' if row.get('token_enabled') else 'native_or_fallback'].append((client, row))

    pairs = {
        'coordinator_received_to_E_submit_ms': ('received', 'E_submit'),
        'E_HTTP_ms': ('E_submit', 'E_complete'),
        'E_complete_to_P_submit_ms': ('E_complete', 'P_submit'),
        'P_HTTP_ms': ('P_submit', 'P_ready'),
        'P_ready_to_D_submit_ms': ('P_ready', 'D_submit'),
        'D_submit_to_first_proxy_generated_token_ms': ('D_submit', 'D_first_generated_token'),
        'D_submit_to_first_SSE_bytes_ms': ('D_submit', 'D_first_SSE_bytes'),
        'D_HTTP_request_ms': ('D_submit', 'D_http_complete'),
    }
    result = {}
    for name, rows in groups.items():
        facts = [row['D_at_P_ready'] for _, row in rows if row.get('D_at_P_ready') is not None]
        no_active = sum(fact['active_http_requests'] == 0 for fact in facts)
        quiet = sum(fact['active_http_requests'] == fact['pending_admissions'] == 0
                    and fact['suffix_owner'] is None for fact in facts)
        result[name] = {
            'matched': len(rows), 'P_ready_observed_D_HTTP_snapshots': len(facts),
            'P_ready_no_active_D_HTTP_fraction': no_active / len(facts) if facts else None,
            'P_ready_no_active_pending_or_suffix_D_HTTP_fraction': quiet / len(facts) if facts else None,
            'client_TTFT_ms': distribution([client['TTFT_ms'] for client, _ in rows]),
            'P_ready_active_D_HTTP': distribution([fact['active_http_requests'] for fact in facts]),
            'P_ready_pending_D_HTTP': distribution([fact['pending_admissions'] for fact in facts]),
            'suffix_tokens': distribution([row['token_partition']['suffix_tokens'] for _, row in rows
                                           if row.get('token_partition')]),
            **{metric: distribution([(row[end] - row[start]) * 1000 for _, row in rows
                                     if start in row and end in row and row[end] >= row[start]])
               for metric, (start, end) in pairs.items()},
        }
    return {'groups': result,
            'scope': 'P-ready is full P ready for Native or actual prefix ready for a selected split. '
                     'D state is HTTP admission and cannot establish actual kernel/Suffix isolation. '
                     'D submit to first token includes imports, suffix if any, sampling and delivery.'}


def effect(before, observed, after):
    result = {}
    for key in COMPARISON_METRICS:
        a, b, c = before.get(key), observed.get(key), after.get(key)
        center = (a + c) / 2 if a is not None and c is not None else None
        result[key] = {'clean_before': a, 'profile': b, 'clean_after': c, 'clean_mean': center,
                       'profile_ratio_to_clean': b / center if center and b is not None else None,
                       'clean_after_ratio_to_before': c / a if a and c is not None else None}
    return result


def evaluate(trace_path, capacity_path=None):
    trace_path = Path(trace_path)
    tag = trace_path.name.removesuffix('_trace.json')
    run = trace_path.parent.parent
    capacity_path = Path(capacity_path) if capacity_path is not None else run / 'analysis/capacity_prefill_free.json'
    gate = discovery_gate(trace_path, capacity_path)
    trace = read(trace_path)
    configs = read(run / 'configs' / f'{tag}_profile.json')
    phase_metrics = {phase: read(run / 'analysis' / f'{tag}_{phase}_metrics.json') for phase in CPC_PHASES}
    profile_rows, proxy = paired_rows(run, f'{tag}_profile')
    raw = {phase: jsonl(run / 'raw' / f'{tag}_{phase}_requests.jsonl')
           if phase != 'profile' else profile_rows for phase in CPC_PHASES}
    slos = {phase: decode_slo(raw[phase], phase_metrics[phase]['planned_count']) for phase in CPC_PHASES}
    left, right = trace['window_monotonic']
    p_busy, d_busy = (trace['kernel_intervals'][role] for role in ('prefill', 'decode'))
    p_waiting, _ = scheduler_intervals(run / 'raw/profiles' / f'{tag}_profile' / 'prefill', left, right)
    work = [(row['P_ready'], row.get('D_http_complete', row['finished']))
            for row in proxy if row['status'] == 'MEASURED_ONLINE']
    physical = complement(d_busy, left, right)
    robust = complement(merged(d_busy + work, left, right), left, right)
    p_busy_waiting = intersect(p_busy, p_waiting)
    window = [{'threshold_ms': threshold,
               'D_physical_idle': coverage(physical, p_busy, p_busy_waiting, right - left, threshold),
               'D_robust_quiet': coverage(robust, p_busy, p_busy_waiting, right - left, threshold)}
              for threshold in WINDOW_MS]
    env_path = run / 'state/environment.json'
    env = read(env_path) if env_path.is_file() else {}
    argv = env.get('argv', [])
    workload = configs['workload']
    return {**gate, 'run': run.name, 'tag': tag, 'trace': str(trace_path),
            'capacity_reference': str(capacity_path) if capacity_path.is_file() else None,
            'decode_cap': workload.get('decode_cap', '1' if gate['batch_one'] else 'free'),
            'method': workload.get('method'), 'output_mode': configs.get('output_mode'),
            'arrival_seed': configs.get('seed'), 'arrival_SHA256': configs.get('arrival_SHA256'),
            'manifest_SHA256': workload.get('manifest_SHA256'),
            'other_GPU_burners_setting': 'off' if '--all-burners-off' in argv else 'unchanged_outside_group',
            'capture_setting': 'short_ROI' if '--short-roi' in argv else 'full_cohort',
            'evidence': trace['evidence'], 'duration_seconds': right - left,
            'p_busy_seconds': elapsed(p_busy), 'kernel_busy_fraction': trace['kernel_busy_fraction'],
            'clean_decode_SLO': {phase: slos[phase] for phase in ('clean_before', 'clean_after')},
            'profile_decode_SLO': slos['profile'],
            'CPC_effect': effect(*(phase_metrics[phase] for phase in CPC_PHASES)),
            'idle_window_sensitivity': window, 'runtime_stages': runtime_stages(profile_rows, proxy)}


def matched_conditions(profiles):
    keys = ('qps', 'decode_cap', 'method', 'main100_cycles', 'output_mode', 'arrival_SHA256', 'manifest_SHA256')
    comparisons = []
    for a, b in itertools.combinations(profiles, 2):
        if any(a[key] != b[key] for key in keys) or not a['arrival_SHA256'] or not a['manifest_SHA256']:
            continue
        changed = [key for key in ('other_GPU_burners_setting', 'capture_setting') if a[key] != b[key]]
        if not changed:
            continue
        metrics = {}
        for key in COMPARISON_METRICS:
            x, y = a['CPC_effect'][key]['clean_mean'], b['CPC_effect'][key]['clean_mean']
            metrics[key] = {'first_clean_mean': x, 'second_clean_mean': y,
                            'second_to_first_clean_ratio': y / x if x and y is not None else None}
        comparisons.append({'first': a['run'], 'second': b['run'], 'differing_conditions': changed,
                            'metrics': metrics, 'scope': 'Descriptive matched workload summaries. No paired-start CI.'})
    return comparisons


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--trace', type=Path, nargs='+', required=True)
    parser.add_argument('--capacity', type=Path)
    parser.add_argument('--cohort', action='append', nargs=2, default=[], metavar=('RUN_DIR', 'TAG'))
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    profiles = [evaluate(path, args.capacity) for path in args.trace]
    cohorts = [{'run': run, 'tag': tag, **runtime_stages(*paired_rows(Path(run), tag))} for run, tag in args.cohort]
    value = {'profiles': profiles, 'matched_condition_comparisons': matched_conditions(profiles), 'cohorts': cohorts}
    if args.output:
        with args.output.open('x') as stream:
            json.dump(value, stream, separators=(',', ':'))
            stream.write('\n')
    else:
        print(json.dumps(value, indent=2))


if __name__ == '__main__':
    main()
