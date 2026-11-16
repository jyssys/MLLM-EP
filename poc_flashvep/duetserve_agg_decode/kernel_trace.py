"""Reconstruct actual Nsight kernel unions without summing overlapping streams."""
import argparse
import collections
import json
import re
import sqlite3
import statistics
import subprocess

from .common import ROOT, write
from .interference import overhead
from .statistics import percentile


def union(intervals, lower, upper):
    result = []
    for start, end in sorted((max(start, lower), min(end, upper)) for start, end in intervals
                             if end > lower and start < upper):
        if result and start <= result[-1][1]:
            result[-1][1] = max(result[-1][1], end)
        else:
            result.append([start, end])
    return result


def duration(intervals):
    return sum(end - start for start, end in intervals)


def gaps(intervals, lower, upper):
    cursor, result = lower, []
    for start, end in intervals:
        if start > cursor:
            result.append([cursor, start])
        cursor = max(cursor, end)
    if cursor < upper:
        result.append([cursor, upper])
    return result


def analyze(name):
    folder = ROOT / 'raw' / name
    report = next(folder.glob('nsys_*.nsys-rep'))
    role = report.stem.removeprefix('nsys_')
    target = report.with_suffix('.sqlite')
    assert report.exists()
    if not target.exists():
        with (folder / 'nsys_sqlite_export.log').open('w') as stream:
            subprocess.run(['/usr/local/bin/nsys', 'export', '--type', 'sqlite', '--output', str(target), str(report)],
                           stdout=stream, stderr=subprocess.STDOUT, check=True)
    db = sqlite3.connect('file:' + str(target) + '?mode=ro', uri=True)
    db.row_factory = sqlite3.Row
    markers = [dict(row) for row in db.execute('SELECT start,end,text FROM NVTX_EVENTS WHERE text LIKE ?', ('ELASTIC_MONO_%',))]
    start = next(row for row in markers if row['text'].startswith('ELASTIC_MONO_START_'))
    end = next(row for row in markers if row['text'].startswith('ELASTIC_MONO_END_'))
    lower, upper = start['end'], end['start']
    assert upper > lower
    clock = json.loads((folder / f'nsys_clock_{role}.json').read_text())
    offsets = [(clock[label]['before'] - row['start'] / 1e9,
                clock[label]['after'] - row['start'] / 1e9)
               for label, row in (('start', start), ('end', end))]
    offset = statistics.mean((a + b) / 2 for a, b in offsets)
    # Bound alignment error by both marker brackets and their measured drift.
    alignment_error_ms = 1000 * max(abs(bound - offset) for pair in offsets for bound in pair)
    kernels = [dict(row) for row in db.execute('''SELECT k.start,k.end,k.deviceId,k.streamId,
        k.correlationId,k.graphNodeId,k.launchType,s.value AS name FROM CUPTI_ACTIVITY_KIND_KERNEL k
        JOIN StringIds s ON s.id=k.demangledName WHERE k.end>? AND k.start<?''', (lower, upper))]
    assert kernels and len({row['deviceId'] for row in kernels}) == 1
    copies = [dict(row) for row in db.execute('''SELECT start,end,streamId,bytes,copyKind FROM
        CUPTI_ACTIVITY_KIND_MEMCPY WHERE end>? AND start<?''', (lower, upper))]
    diagnostics = [dict(row) for row in db.execute('SELECT * FROM DIAGNOSTIC_EVENT')]
    serious = [row for row in diagnostics if re.search(r'drop|lost|incomplete|error|fail', row['text'], re.I)]
    compute = [row for row in kernels if not re.search('nccl|rccl', row['name'], re.I)]
    comm = [row for row in kernels if re.search('nccl|rccl', row['name'], re.I)]
    all_union = union([(r['start'], r['end']) for r in kernels], lower, upper)
    compute_union = union([(r['start'], r['end']) for r in compute], lower, upper)
    comm_union = union([(r['start'], r['end']) for r in comm], lower, upper)
    inactive = gaps(all_union, lower, upper)
    idle_ms = [(b - a) / 1e6 for a, b in inactive]
    seconds = (upper - lower) / 1e9
    prefix = name.rsplit('_NSYS_', 1)[0]
    gate = overhead(prefix)
    overhead_pass = gate.get('comparisons', {}).get('NSYS', {}).get('qualified', False)
    receipt = json.loads((folder / 'dispatcher_result.json').read_text())
    integrity = not serious and alignment_error_ms <= 1.0 and bool(kernels)
    # Actual node records demonstrate graph coverage. A graph replay itself
    # is not an extra kernel interval and must not be counted twice.
    graph_nodes = sum(bool(row['graphNodeId']) for row in kernels)
    graph_launches = db.execute('''SELECT count(*) FROM CUPTI_ACTIVITY_KIND_RUNTIME r
        JOIN StringIds s ON s.id=r.nameId WHERE r.end>? AND r.start<? AND s.value LIKE ?''',
        (lower, upper, '%GraphLaunch%')).fetchone()[0]
    graph_coverage = not graph_launches or graph_nodes > 0
    qualified = bool(receipt['complete'] and overhead_pass and integrity and graph_coverage)
    copy_stats = {}
    copy_labels = {row['id']: row['label'] for row in db.execute('SELECT id,label FROM ENUM_CUDA_MEMCPY_OPER')}
    for kind in sorted({r['copyKind'] for r in copies}):
        selected = [r for r in copies if r['copyKind'] == kind]
        intervals = union([(r['start'], r['end']) for r in selected], lower, upper)
        copy_stats[str(kind)] = dict(label=copy_labels[kind], count=len(selected), bytes=sum(r['bytes'] for r in selected),
                                    union_ms=duration(intervals) / 1e6)
    result = dict(name=name, role=role, physical_GPU=0, evidence='MEASURED' if qualified else 'TRACE_ANALYSIS',
        clean_physical_kernel_measurement='QUALIFIED_SHORT_WINDOW' if qualified else 'NOT_MEASURED',
        diagnostic_values=dict(window_seconds=seconds, kernel_count=len(kernels),
            kernel_union_busy_seconds=duration(all_union) / 1e9,
            kernel_union_busy_percent=100 * duration(all_union) / (upper - lower),
            kernel_inactive_seconds=sum(idle_ms) / 1000,
            kernel_inactive_percent=100 * sum(idle_ms) / (1000 * seconds),
            compute_union_ms=duration(compute_union) / 1e6, communication_kernel_union_ms=duration(comm_union) / 1e6,
            idle_window_count=len(inactive), idle_mean_ms=statistics.mean(idle_ms) if idle_ms else None,
            idle_p50_ms=percentile(idle_ms, .5), idle_p95_ms=percentile(idle_ms, .95),
            idle_p99_ms=percentile(idle_ms, .99), idle_max_ms=max(idle_ms, default=0),
            idle_thresholds={str(t): dict(count=sum(v >= t for v in idle_ms),
                total_ms=sum(v for v in idle_ms if v >= t)) for t in (1, 2, 5, 10, 20, 30, 50, 100)},
            graph_node_kernel_records=graph_nodes, graph_launches=graph_launches,
            graph_coverage_pass=graph_coverage, stream_counts=dict(collections.Counter(r['streamId'] for r in kernels)),
            memcpy_kind=copy_stats),
        overhead_pass=overhead_pass, dispatcher_complete=receipt['complete'], collector_integrity_pass=integrity,
        diagnostics=diagnostics, clock_alignment_error_bound_ms=alignment_error_ms,
        trace_window_ns=[lower, upper], trace_to_host_offset_seconds=offset,
        scope=f'Actual Nsight/CUPTI activity on {role}/GPU0 only. Union across all recorded streams. Copies remain separate. Kernel inactivity is not Decode-safe compute capacity. A2/A1 and matched-token overhead gates must pass. No other run is combined into this physical timeline.')
    write(ROOT / 'analysis' / (name + '_kernel_union.json'), result)
    write(ROOT / 'analysis' / (name + '_kernel_intervals.json'), dict(kernels=kernels,
        copies=copies, idle_intervals_ns=inactive, markers=markers,
        trace_to_host_offset_seconds=offset, clock_alignment_error_bound_ms=alignment_error_ms))
    print(name, qualified, result['diagnostic_values']['kernel_union_busy_percent'], alignment_error_ms)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('names', nargs='+')
    for name in parser.parse_args().names:
        analyze(name)
