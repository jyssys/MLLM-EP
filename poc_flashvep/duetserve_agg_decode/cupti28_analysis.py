"""Separate API28 kernel records from matched serving-interference gates."""
import csv
import fcntl
import json
import statistics

from .common import QUIET_LOCK, ROOT, resources, write
from .interference import METRICS, intervals
from .kernel_trace import duration, gaps, union
from .statistics import percentile


def read(path):
    return json.loads(path.read_text())


def kernel_window(folder, role):
    bounded = folder / f'activity_{role}_background.json'
    if bounded.exists():
        value = read(bounded)
        a, b = value['start'], value['end']
        left, right = a['cupti_ns'] + 100_000_000, b['cupti_ns'] - 100_000_000
        lower = a['host_midpoint'] + .1
        upper = b['host_midpoint'] - .1
        capture_mode = 'BOUNDED_BACKGROUND'
    else:
        value = read(folder / f'activity_{role}_end.json')
        a = read(folder / f'activity_{role}_begin.json')['anchor']
        b = value['anchor']
        measured = read(folder / 'measurement_begin.json')['time']
        lower, upper = measured + 10.1, measured + 29.9
        scale = (b['cupti_ns'] - a['cupti_ns']) / (b['host_midpoint'] - a['host_midpoint'])
        left = int(a['cupti_ns'] + (lower - a['host_midpoint']) * scale)
        right = int(a['cupti_ns'] + (upper - a['host_midpoint']) * scale)
        capture_mode = 'PRECONTEXT_FULL_RECORDING_SHORT_ANALYSIS_WINDOW'
    kernels = []
    for row in csv.DictReader((folder / f'activity_{role}.csv').open()):
        if row['kind'] == 'kernel' and int(row['end_ns']) > left and int(row['start_ns']) < right:
            kernels.append(row)
    busy = union([(int(r['start_ns']), int(r['end_ns'])) for r in kernels], left, right)
    idle = gaps(busy, left, right)
    lengths = [(y - x) / 1e6 for x, y in idle]
    total_ns = right - left
    counters = value['counters']
    graph_count = sum(int(r['graph_node']) != 0 for r in kernels)
    graph_needed = role != 'prefill'
    integrity = (value['result'] == 0 and counters['dropped'] == 0 and counters['invalid'] == 0
                 and bool(kernels) and (not graph_needed or graph_count > 0))
    result = dict(role=role, result=value['result'], counters=counters, capture_mode=capture_mode,
        capture_begin_host=lower, capture_end_host=upper, window_seconds=total_ns / 1e9,
        host_CUPTI_anchor_half_width_ms=max(a['host_upper'] - a['host_lower'], b['host_upper'] - b['host_lower']) * 500,
        kernel_records=len(kernels), graph_kernel_records=graph_count,
        stream_ids=sorted({int(r['stream']) for r in kernels}),
        busy_seconds=duration(busy) / 1e9,
        busy_percent=100 * duration(busy) / total_ns,
        idle_seconds=duration(idle) / 1e9,
        idle_percent=100 * duration(idle) / total_ns,
        idle_window_count=len(idle), idle_mean_ms=statistics.mean(lengths) if lengths else 0,
        idle_p50_ms=percentile(lengths, .5), idle_p95_ms=percentile(lengths, .95),
        idle_p99_ms=percentile(lengths, .99), idle_max_ms=max(lengths, default=0),
        thresholds={str(t): dict(count=sum(x >= t for x in lengths),
            total_ms=sum(x for x in lengths if x >= t)) for t in (1, 2, 5, 10, 20, 30, 50, 100)},
        integrity_passed=integrity, idle_windows_CUPTI_ns=idle,
        scope='Union of actual concurrent kernel intervals across all streams in the trimmed short window. Copies are intentionally not traced. Kernel inactivity is not decode-safe extra-compute capacity.')
    return result


def qualify(method):
    with QUIET_LOCK.open('a+') as lease:
        fcntl.flock(lease, fcntl.LOCK_EX)
        owner = resources()
        if not owner.processes():
            owner.restore(list(range(8)))
        return analyze(method)


def analyze(method):
    names = {s: f'api28_{method}_N128_O200_q6_s2026111641_{s}_v1' for s in ('A1', 'CUPTI', 'A2')}
    retry = names['CUPTI'].removesuffix('_v1') + '_v2'
    if (ROOT / 'raw' / retry / 'metrics.json').exists():
        names['CUPTI'] = retry
    metrics = {s: read(ROOT / 'raw' / name / 'metrics.json') for s, name in names.items()}
    controls = {s: metrics[s] for s in ('A1', 'A2')}
    reference = {k: statistics.mean(v[k] for v in controls.values()) for k in METRICS}
    spread = {k: controls['A2'][k] / controls['A1'][k] - 1 for k in METRICS}
    change = {k: metrics['CUPTI'][k] / reference[k] - 1 for k in METRICS}
    folder = ROOT / 'raw' / names['CUPTI']
    roles = ('prefill', 'decode') if method == 'Native' else ('replica0', 'replica1')
    windows = {r: kernel_window(folder, r) for r in roles}
    interval_sets = {s: intervals(name) for s, name in names.items()}
    roi = {}
    for role, window in windows.items():
        actual = interval_sets['CUPTI']
        keys = [k for k, v in actual.items() if window['capture_begin_host'] <= v['start']
            and v['end'] <= window['capture_end_host']
            and (method == 'Native' or k[0] % 2 == roles.index(role))
            and all(k in interval_sets[s] for s in controls)]
        values = {}
        for scope in ('parent_ms', 'worker_ms'):
            if not keys:
                continue
            observed = [actual[k][scope] for k in keys]
            paired = {s: [interval_sets[s][k][scope] for k in keys] for s in controls}
            for label, op in (('mean', statistics.mean), ('p99', lambda x: percentile(x, .99))):
                expected = statistics.mean(op(paired[s]) for s in controls)
                values[scope + '_' + label] = dict(actual=op(observed), control_mean=expected,
                    relative_change=op(observed) / expected - 1,
                    A2_A1_relative_change=op(paired['A2']) / op(paired['A1']) - 1)
        roi[role] = dict(matched_intervals=len(keys), metrics=values,
            passed=bool(keys) and all(abs(v['relative_change']) <= .05 and
                abs(v['A2_A1_relative_change']) <= .05 for v in values.values()))
    control_pass = all(abs(x) <= .05 for x in spread.values())
    whole_pass = all(abs(x) <= .05 for x in change.values())
    passed = control_pass and whole_pass and all(r['passed'] for r in roi.values()) \
        and all(r['integrity_passed'] for r in windows.values())
    status = 'QUALIFIED_SHORT_WINDOW' if passed else 'NOT_MEASURED_CLEAN_PHYSICAL_CAPACITY'
    result = dict(method=method, names=names, qualified=passed, status=status,
        whole_run_control_A2_A1_relative_change=spread, whole_run_relative_change=change,
        control_passed=control_pass, whole_run_passed=whole_pass, ROI=roi,
        role_windows=windows, evidence='MEASURED' if passed else 'TRACE_ANALYSIS',
        scope='A predeclared matched A/activity/A check. Every whole-run and matched-interval change must stay within5%. The collector witness validates eager/graph record coverage only. A qualified short window is not a whole-run utilization or request-feasible offloading-capacity claim.')
    write(ROOT / 'analysis' / ('api28_' + method + '_serving_gate.json'), result)
    print('API28_SERVING_GATE', method, status, flush=True)
    return result


if __name__ == '__main__':
    for selected in ('Native', 'DP2_B8192_S1024'):
        qualify(selected)
