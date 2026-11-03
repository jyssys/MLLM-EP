"""Describe kernel names without treating overlapping sums as E2E shares."""
import argparse
import collections
import csv
from .common import *
from .analysis_lane import quiet_analysis


def category(name):
    low = name.lower()
    if 'flash_attn' in low or 'flashattention' in low:
        return 'attention_name_match'
    if any(word in low for word in ('nvjet', 'gemm', 'matmul')):
        return 'GEMM_name_match'
    if any(word in low for word in ('copy', 'transfer', 'index_put', 'index_elementwise', 'catarray', 'fill')):
        return 'copy_index_or_fill_name_match'
    if any(word in low for word in ('elementwise', 'reduce', 'vectorized')):
        return 'elementwise_or_reduction_name_match'
    return 'other_kernel_names'


def analyze(names):
    summaries = []
    for name in names:
        folder = ROOT/'raw'/name
        gates = json.loads((folder/'submission_queue_audit.json').read_text())
        left, right = gates['observation_start'], gates['observation_end']
        kernels = json.loads((folder/'aligned_kernels.json').read_text())['kernels']
        by_name = collections.defaultdict(lambda: [0, 0.])
        for row in kernels:
            duration = max(0., min(right, row['end_monotonic'])-max(left, row['start_monotonic']))
            if duration:
                value = by_name[row['name']]
                value[0] += 1
                value[1] += duration
        total = sum(value[1] for value in by_name.values())
        rows = [dict(kernel=key, category=category(key), invocations=value[0], summed_device_seconds=value[1],
            share_summed_kernel_duration_percent=100*value[1]/total) for key, value in by_name.items()]
        rows.sort(key=lambda row: row['summed_device_seconds'], reverse=True)
        with (folder/'kernel_name_duration_mix.csv').open('w', newline='') as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        groups = collections.defaultdict(lambda: [0, 0.])
        for row in rows:
            groups[row['category']][0] += row['invocations']
            groups[row['category']][1] += row['summed_device_seconds']
        summary = dict(name=name, observation_start=left, observation_end=right,
            observation_seconds=right-left, summed_kernel_seconds=total,
            categories={key: dict(invocations=value[0], summed_device_seconds=value[1],
                share_summed_kernel_duration_percent=100*value[1]/total) for key, value in groups.items()},
            scope='Heuristic kernel-name categories. Overlapping durations are summed, not unioned. These percentages are not E2E shares, SM occupancy, achieved FLOP/s, or a physical-resource saturation proof.')
        write(folder/'kernel_name_duration_mix.json', summary)
        summaries.append(summary)
        del kernels
    write(ROOT/'kernel_name_duration_mix_summary.json', summaries)
    print(json.dumps(summaries), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('names', nargs='+')
    args = parser.parse_args()
    with quiet_analysis('kernel_name_duration_mix'):
        analyze(args.names)
