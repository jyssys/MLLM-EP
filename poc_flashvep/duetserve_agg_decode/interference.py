"""Qualify diagnostic overhead and describe observed batch composition."""
import json
import statistics

from .analyze import records
from .common import ROOT, write
from .statistics import percentile


METRICS = ('mean_TBT_ms', 'TTFT_p95_ms', 'ITL_p99_ms', 'TPOT_p95_ms', 'output_tokens_s')


def decode_output_frontier(start, count, known_prefill_len, original_prompt_len=8000):
    if start + count < known_prefill_len:
        return None  # A partial recomputation does not emit an output token.
    return int(start + count - original_prompt_len + 1)


def intervals(name):
    result = {}
    for row in records(ROOT / 'raw' / name / 'requests.jsonl'):
        for a, b in zip(row['token_events'], row['token_events'][1:]):
            assert b['count'] - a['count'] == 1
            result[(row['sample_id'], b['count'])] = dict(
                parent_ms=1000 * (b['time'] - a['time']),
                worker_ms=1000 * (b['worker_time'] - a['worker_time']),
                start=a['worker_time'], end=b['worker_time'])
    return result


def overhead(prefix):
    names = {label: prefix + '_' + label + '_v1' for label in ('A1', 'A2', 'HOST_ONLY', 'HOST_EVENT', 'NSYS')}
    for label, name in list(names.items()):
        retry = name.removesuffix('_v1') + '_v2'
        if (ROOT / 'raw' / retry / 'metrics.json').exists():
            names[label] = retry
    controls = {}
    for label in ('A1', 'A2'):
        path = ROOT / 'raw' / names[label] / 'metrics.json'
        if not path.exists():
            return dict(prefix=prefix, qualified=False, status='CONTROLS_INCOMPLETE')
        controls[label] = json.loads(path.read_text())
    reference = {key: statistics.mean(controls[label][key] for label in controls) for key in METRICS}
    spread = {key: abs(controls['A2'][key] / controls['A1'][key] - 1) for key in METRICS}
    control_pass = all(value <= .05 for value in spread.values())
    a_intervals = {label: intervals(names[label]) for label in controls}
    comparisons = {}
    for label in ('HOST_ONLY', 'HOST_EVENT', 'NSYS'):
        folder = ROOT / 'raw' / names[label]
        if not (folder / 'metrics.json').exists():
            comparisons[label] = dict(qualified=False, status='NOT_RUN_OR_FAILED')
            continue
        metric = json.loads((folder / 'metrics.json').read_text())
        errors = {key: metric[key] / reference[key] - 1 for key in METRICS}
        anchor = json.loads((folder / 'measurement_begin.json').read_text())['time']
        begin, end = anchor + 10, anchor + 12
        clock = next(folder.glob('nsys_clock_*.json'), folder / 'nsys_clock_replica0.json')
        if clock.exists():
            value = json.loads(clock.read_text())
            if value.get('start') and value.get('end'):
                begin, end = value['start']['after'], value['end']['before']
        observed = intervals(names[label])
        keys = [key for key, value in observed.items()
                if ('Native' in prefix or key[0] % 2 == 0) and value['start'] >= begin and value['end'] < end
                and all(key in a_intervals[a] for a in controls)]
        roi = {}
        for scope in ('parent_ms', 'worker_ms'):
            actual = [observed[key][scope] for key in keys]
            paired = {a: [a_intervals[a][key][scope] for key in keys] for a in controls}
            if not actual:
                continue
            for suffix, operation in (('mean', statistics.mean), ('p99', lambda values: percentile(values, .99))):
                expected = statistics.mean(operation(paired[a]) for a in controls)
                roi[scope + '_' + suffix] = dict(actual=operation(actual), control_mean=expected,
                    relative_change=operation(actual) / expected - 1,
                    A2_A1_relative_change=operation(paired['A2']) / operation(paired['A1']) - 1)
        roi_pass = bool(keys) and all(abs(value['relative_change']) <= .05 and
            abs(value['A2_A1_relative_change']) <= .05 for value in roi.values())
        passed = control_pass and roi_pass and all(abs(value) <= .05 for value in errors.values())
        comparisons[label] = dict(qualified=passed, whole_run_relative_changes=errors,
            matched_ROI_intervals=len(keys), ROI=roi,
            physical_kernel_claim_allowed=False,
            scope='This gate checks interference only. Nsight integrity, graph-node coverage, and clock alignment need separate checks. ROI tokens are not independent server starts.')
    result = dict(prefix=prefix, controls_stable_within_5_percent=control_pass,
        control_relative_spread=spread, comparisons=comparisons,
        evidence='TRACE_ANALYSIS',
        scope='Two marker-free controls. Require absolute changes at most5% in whole-run metrics and matched-interval ROI metrics. Passing is not a production overhead guarantee.')
    write(ROOT / 'analysis' / (prefix + '_interference_gate.json'), result)
    return result


def composition(name):
    folder = ROOT / 'raw' / name
    all_intervals = intervals(name)
    labels = {}
    forward = []
    for path in folder.glob('iteration_trace_*.json'):
        trace = json.loads(path.read_text())
        for item in trace['execution']:
            if item['kind'] != 'MODEL_FORWARD':
                continue
            if not any(rid.split('-')[0].startswith('q') and rid.split('-')[0][1:].isdigit()
                       for rid in item['request_ids']):
                continue
            prefill = sum(max(0, min(count, length - start))
                          for start, count, length in zip(item['computed_tokens'], item['query_lengths'], item['prefill_lens']))
            forward.append(dict(prefill_rows=prefill, requests=len(item['request_ids']),
                query_rows=item['query_tokens'], CUDA_event_enclosure_ms=item['CUDA_event_enclosure_ms'],
                host_enclosure_ms=1000 * (item['host_end'] - item['host_start']),
                existing_decode_rows=sum(count for count, flag in zip(item['query_lengths'], item['prefilling']) if not flag),
                contains_recomputed_generated_prefix=any(length > 8000 for length in item['prefill_lens'])))
            for rid, start, count, length in zip(item['request_ids'], item['computed_tokens'], item['query_lengths'], item['prefill_lens']):
                external = rid.split('-')[0]
                if not (external.startswith('q') and external[1:].isdigit()):
                    continue
                output_count = decode_output_frontier(start, count, length)
                if output_count is not None and output_count >= 2:
                    key = (int(external[1:]), output_count)
                    assert key not in labels, 'Duplicate actual Decode output frontier'
                    labels[key] = ('PREFILL_PRESENT' if prefill else 'DECODE_ONLY')
    classes = {}
    for label in ('PREFILL_PRESENT', 'DECODE_ONLY', 'UNMATCHED'):
        values = [v for key, v in all_intervals.items() if labels.get(key, 'UNMATCHED') == label]
        classes[label] = dict(intervals=len(values), fraction=len(values) / len(all_intervals),
            mean_parent_ms=statistics.mean(v['parent_ms'] for v in values) if values else None,
            mean_worker_ms=statistics.mean(v['worker_ms'] for v in values) if values else None,
            p99_parent_ms=percentile([v['parent_ms'] for v in values], .99))
    result = dict(name=name, classes=classes, forward_observations=forward,
        actual_interval_label_coverage=1 - classes['UNMATCHED']['fraction'],
        evidence='TRACE_ANALYSIS',
        scope='Join actual model input frontiers to output counts. PREFILL_PRESENT includes KV-pressure recomputation, not only a new-prompt/decode mixed batch. Descriptive class means are not exclusive GPU-causal contributions. CUDA-event enclosure is not kernel time.')
    write(ROOT / 'analysis' / (name + '_composition.json'), result)
    return result


def mixture_change(low, high):
    lo, hi = composition(low), composition(high)
    mixed, decode = 'PREFILL_PRESENT', 'DECODE_ONLY'
    f0, f1 = lo['classes'][mixed]['fraction'], hi['classes'][mixed]['fraction']
    m0, m1 = lo['classes'][mixed]['mean_parent_ms'], hi['classes'][mixed]['mean_parent_ms']
    d0, d1 = lo['classes'][decode]['mean_parent_ms'], hi['classes'][decode]['mean_parent_ms']
    assert lo['actual_interval_label_coverage'] == hi['actual_interval_label_coverage'] == 1
    changes = dict(mixed_frequency_ms=(f1 - f0) * ((m0 + m1) / 2 - (d0 + d1) / 2),
        mixed_class_latency_ms=(f0 + f1) / 2 * (m1 - m0),
        decode_only_class_latency_ms=(1 - (f0 + f1) / 2) * (d1 - d0))
    total = (f1 * m1 + (1 - f1) * d1) - (f0 * m0 + (1 - f0) * d0)
    assert abs(sum(changes.values()) - total) < 1e-9
    result = dict(low=low, high=high, measured_mean_TBT_change_ms=total,
        symmetric_mixture_identity=changes, fractions={k: v / total for k, v in changes.items()},
        evidence='TRACE_ANALYSIS',
        scope='An exact descriptive identity of the measured interval mixture. The historical mixed_* keys mean PREFILL_PRESENT, including recomputation. Contributions are not exclusive GPU-causal effects. Decode-only intervals can wait behind an earlier mixed iteration. SM/HBM/cache/launch causal shares remain NOT_IDENTIFIED.')
    write(ROOT / 'analysis/large_budget_QPS_mixture_decomposition.json', result)
    return result


if __name__ == '__main__':
    prefixes = [f'profile_DP2_B{budget}_S{slots}_FULL_N128_O200_q{qps}_s2026111616'
                for budget, slots, qps in ((8192, 1024, 2), (8192, 1024, 6), (512, 16, 6))]
    for prefix in prefixes:
        result = overhead(prefix)
        print(prefix, result.get('comparisons', {}))
        for label in ('HOST_ONLY', 'HOST_EVENT', 'NSYS'):
            name = prefix + '_' + label + '_v1'
            if list((ROOT / 'raw' / name).glob('iteration_trace_*.json')):
                composition(name)
