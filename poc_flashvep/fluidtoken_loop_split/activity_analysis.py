"""Align direct CUPTI kernel timestamps to the same-run host timeline."""
import argparse
import csv
from .common import *


def analyze(name, clean, role='prefill'):
    configure()
    assert role in ('prefill', 'decode')
    suffix = '_decode' if role == 'decode' else ''
    folder = ROOT / 'raw' / name
    capture = json.loads((folder / ('activity_capture' + suffix + '.json')).read_text())
    assert capture['role'] == role
    calibration = json.loads((ROOT / 'raw/activity_only_collector_selftest/activity_selftest.json').read_text())
    assert calibration['passed'], 'Collector coverage self-test is mandatory'
    validation = json.loads((ROOT / 'tools/activity_validation.json').read_text())
    assert validation['selftest_SHA256'] == digest(ROOT / 'raw/activity_only_collector_selftest/activity_selftest.json')
    assert validation['collector_build']['library_SHA256'] == capture['collector_build']['library_SHA256']
    assert capture['status'] == 0 and capture['dropped'] == capture['invalid'] == 0
    start, stop = capture['start'], capture['stop']
    left, right = start['device_clock_ns'], stop['device_clock_ns']
    origin = start['monotonic_s']
    duration = (right - left) / 1e9
    # A second independent host bracket checks the timestamp mapping.
    mapping_error = abs(origin + duration - stop['monotonic_s'])
    tolerance = start['host_uncertainty_s'] + stop['host_uncertainty_s'] + .001
    assert mapping_error <= tolerance, 'CUPTI to host clock mapping failed'
    rows = list(csv.DictReader((folder / ('kernel_activity' + suffix + '.csv')).open()))
    intervals, kernels = [], []
    for row in rows:
        a, b = int(row['start_ns']), int(row['end_ns'])
        if b <= left or a >= right:
            continue
        intervals.append((max(a, left), min(b, right)))
        kernels.append(dict(start_monotonic=origin + (a - left) / 1e9,
            end_monotonic=origin + (b - left) / 1e9, pid=capture['pid'],
            name='CUPTI_KERNEL_NODE_' + row['graph_node'], device=int(row['device']),
            stream=int(row['stream']), graph_node=int(row['graph_node'])))
    from ..looped_pd_headroom_recovery.profile_analysis import union
    from ..looped_pd_atomic_prefill import evidence
    window = json.loads((folder / 'window.json').read_text())
    baseline = json.loads((ROOT / 'raw' / clean / 'window.json').read_text())
    ratio = (window['end'] - window['start']) / (baseline['end'] - baseline['start']) - 1
    exact = evidence.compare(clean, name)['passed']
    result = dict(name=name, clean=clean, role=role, status='PARSED', bounded_capture=True,
        recorded_kernel_invocations=len(kernels), window_start_monotonic=origin,
        window_end_monotonic=origin + duration, window_seconds=duration,
        kernel_busy_fraction=union(intervals) / (right - left),
        anchor_uncertainty_s=start['host_uncertainty_s'], clock_mapping_error_s=mapping_error,
        matched_wall_difference_fraction=ratio, exact_output_parity=exact,
        eligible_for_matched_clean=abs(ratio) <= .05 and exact and bool(kernels),
        evidence='MEASURED_ACTIVITY_ONLY_CUPTI_KERNEL_INTERVALS',
        scope='Actual kernel union with zero dropped or invalid records. Not SM occupancy. Same-run aligned gates remain mandatory.')
    write(folder / ('aligned_kernels' + suffix + '.json'), dict(anchor=dict(pid=capture['pid']),
        window_start=origin, window_end=origin + duration, kernels=kernels,
        evidence='MEASURED_ACTIVITY_ONLY_CUPTI_KERNEL_INTERVALS'))
    write(folder / ('profile_analysis' + suffix + '.json'), result)
    return result


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('name')
    p.add_argument('clean')
    args = p.parse_args()
    print(json.dumps(analyze(args.name, args.clean)))
