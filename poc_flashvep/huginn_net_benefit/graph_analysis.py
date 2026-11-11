"""Measure activity-cover gaps. Graph intervals do not reveal internal kernel gaps."""
import argparse
import sqlite3
from .common import *
from .cost_analysis import quantiles


def union(intervals, left, right):
    result = []
    for a, b in sorted((max(a, left), min(b, right)) for a, b in intervals if b > left and a < right):
        if b <= a:
            continue
        if result and a <= result[-1][1]:
            result[-1] = (result[-1][0], max(b, result[-1][1]))
        else:
            result.append((a, b))
    return result


def analyze(name):
    folder = ROOT / 'raw' / name
    db = sqlite3.connect(folder / 'graph_enclosures.sqlite')
    clocks = [json.loads(p.read_text()) for p in folder.glob('graph_trace_clock_*.json')]
    start = next(row for value in clocks for row in value['records'] if row['event'] == 'START')
    anchor = db.execute('select start from NVTX_EVENTS where text=?', (start['nvtx_name'],)).fetchone()[0]
    offset = start['monotonic'] - anchor / 1e9
    cfg = json.loads((folder / 'configuration.json').read_text())
    window = json.loads((folder / 'window.json').read_text())
    # Start/stop commands can perturb the edges. Analyze only fully active time.
    lower = max(window['start'], start['upper'])
    upper = min([window['end']] + [row['lower'] for value in clocks
        for row in value['records'] if row['event'] == 'STOP'])
    mapping = json.loads((folder / 'worker_map.json').read_text())
    result = dict(name=name, lower=lower, upper=upper, seconds=upper-lower, roles={},
        measurement_scope='GPU graph enclosures plus explicit kernels. Their union covers internal graph gaps. Complement gaps form an observed physical compute-idle lower bound, not an exact kernel-idle distribution. This trace also contains profiler interference unless its independent control gate passes.',
        clean_timing_qualified=False)
    for worker in mapping.values():
        device = worker['visible_device']
        compute, copies, counts = [], [], {}
        for table in ('CUPTI_ACTIVITY_KIND_GRAPH_TRACE', 'CUPTI_ACTIVITY_KIND_KERNEL',
                'CUPTI_ACTIVITY_KIND_MEMCPY', 'CUPTI_ACTIVITY_KIND_MEMSET'):
            rows = db.execute(f'select start,end from {table} where deviceId=?', (device,)).fetchall()
            intervals = [(a/1e9+offset, b/1e9+offset) for a, b in rows]
            counts[table] = len(rows)
            (compute if table.endswith(('GRAPH_TRACE', 'KERNEL')) else copies).extend(intervals)
        cover = union(compute, lower, upper)
        occupied = union(compute+copies, lower, upper)
        gaps = []
        cursor = lower
        for a, b in occupied:
            if a > cursor:
                gaps.append((cursor, a))
            cursor = b
        if cursor < upper:
            gaps.append((cursor, upper))
        durations = [1000*(b-a) for a, b in gaps]
        busy = sum(b-a for a, b in cover)
        occupied_s = sum(b-a for a, b in occupied)
        result['roles'][worker['role']] = dict(physical_GPU=worker['physical_GPU'],
            uuid=worker['uuid'], visible_device=device, counts=counts,
            compute_enclosure_union_s=busy, compute_enclosure_union_percent=100*busy/(upper-lower),
            all_activity_enclosure_union_s=occupied_s,
            all_activity_gap_lower_bound_s=(upper-lower)-occupied_s,
            gap_duration_ms=quantiles(durations),
            threshold_windows={str(threshold): dict(count=sum(d >= threshold for d in durations),
                total_ms=sum(d for d in durations if d >= threshold))
                for threshold in (1, 2, 5, 10, 20, 30, 50)},
            activity_enclosures=occupied, gap_lower_bound_intervals=gaps)
    write(folder / 'graph_activity_bounds.json', result)
    db.close()
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('name')
    result = analyze(parser.parse_args().name)
    print(json.dumps({**result, 'roles': {k: {a: b for a, b in v.items()
        if a not in ('activity_enclosures', 'gap_lower_bound_intervals')}
        for k, v in result['roles'].items()}}))
