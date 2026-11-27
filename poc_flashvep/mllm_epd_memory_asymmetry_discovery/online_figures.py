"""Plot online results and HTTP enclosures from the archived JSON/JSONL."""

import argparse
import csv
from pathlib import Path
import re

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from .online_summary import read, rows


STYLE = ('#245a73', '#459474', '#bd713b', '#825980', '#515b6a')


def display(name):
    for word in ('natural149_', 'pressure188_', 'unique91_'):
        name = name.removeprefix(word)
    return re.sub(r'_n(\d+)$', r' · N\1', name).replace('_', ' ')


def write_csv(records, file):
    columns = ['run', 'cohort', 'scheduled', 'completed', 'failed', 'joint_slo_requests', 'joint_goodput_req_s',
               'completion_req_s', 'output_tok_s', 'evidence']
    latency = ('TTFT_p50_ms', 'TTFT_p95_ms', 'TTFT_p99_ms', 'TPOT_p95_ms', 'ITL_p99_ms', 'E2E_p95_ms')
    with file.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=columns + ['QPS', *latency], lineterminator='\n')
        writer.writeheader()
        for row in records:
            payload = {key: row[key] for key in columns}
            payload.update(QPS=row['params']['qps'], **{key: row['latency'][key] for key in latency})
            writer.writerow(payload)


def render_metric(record_set, output, title, n):
    fig, axes = plt.subplots(1, 3, figsize=(11, max(2.7, .4 * len(record_set) + 1.5)), layout='constrained')
    labels = [f"{display(row['run'])} · {row['cohort'].removeprefix('measured_')}" for row in record_set]
    colors = [STYLE[i % len(STYLE)] for i, _ in enumerate(record_set)]
    metrics = [
        ('TTFT p95 (s)', [row['latency']['TTFT_p95_ms'] / 1000 for row in record_set]),
        ('ITL p99 (ms)', [row['latency']['ITL_p99_ms'] for row in record_set]),
        ('Joint SLO goodput (req/s)', [row['joint_goodput_req_s'] for row in record_set]),
    ]
    for i, (axis, (metric, points)) in enumerate(zip(axes, metrics, strict=True)):
        axis.barh(range(len(points)), points, color=colors, height=.62)
        axis.invert_yaxis()
        axis.set(xlabel=metric, yticks=range(len(points)), yticklabels=labels if i == 0 else [])
        axis.grid(axis='x', alpha=.2)
    fig.suptitle(f'{title} · {n} scheduled requests per cohort')
    fig.savefig(output, dpi=160)
    plt.close(fig)


def render_timeline(reference, compare, cohort, output):
    candidate_rows, baseline_rows = [rows(run / 'raw' / 'proxy_epd.jsonl') for run in (compare, reference)]
    valid = lambda row: row.get('status') == 'MEASURED_ONLINE' and row['runtime_request_id'].startswith(cohort + '_')
    reference_map = {r['runtime_request_id']: r for r in baseline_rows if valid(r)}
    choice = next(r for r in candidate_rows if valid(r) and r['runtime_request_id'] in reference_map)
    reference_row = reference_map[choice['runtime_request_id']]
    durations = [('E HTTP', 'E_submit', 'E_complete'), ('P HTTP', 'P_submit', 'P_ready'),
                 ('D first SSE', 'D_submit', 'D_first_SSE_bytes'), ('D stream', 'D_first_SSE_bytes', 'D_http_complete')]
    fig, axis = plt.subplots(figsize=(8.6, 2.7), layout='constrained')
    for index, row in enumerate((reference_row, choice)):
        anchor = row['received']
        for k, (name, start, end) in enumerate(durations):
            if start in row and end in row:
                axis.barh(index, row[end] - row[start], left=row[start] - anchor, height=.52,
                          color=STYLE[k], label=name if index == 0 else None)
    axis.set(xlabel='Seconds since proxy receipt', yticks=[0, 1], yticklabels=[display(reference.name), display(compare.name)],
             title=f"Matched request {choice['runtime_request_id']} · HTTP enclosures")
    axis.invert_yaxis()
    axis.legend(frameon=False, loc='upper center', bbox_to_anchor=(.5, -.30), ncol=4)
    fig.savefig(output, dpi=160)
    plt.close(fig)


def render_offered_load(records, output):
    groups = {'NativeStrong': {}, 'Scoped embeddings + local APC': {}}
    for row in records:
        if row['scheduled'] != 149 or row['cohort'] != 'measured_1' or row['failed'] or row['params']['output']:
            continue
        if row['run'].startswith('natural149_native_'):
            group = groups['NativeStrong']
        elif row['run'].startswith('natural149_combined_'):
            group = groups['Scoped embeddings + local APC']
        else:
            continue
        group[row['params']['qps']] = row
    if any(len(values) < 2 for values in groups.values()):
        return
    fig, (ttft, goodput) = plt.subplots(1, 2, figsize=(8.6, 3.6), layout='constrained')
    for index, (name, points) in enumerate(groups.items()):
        qps = sorted(points)
        ttft.plot(qps, [points[x]['latency']['TTFT_p95_ms'] / 1000 for x in qps], 'o-', color=STYLE[index], label=name)
        goodput.plot(qps, [points[x]['joint_goodput_req_s'] for x in qps], 'o-', color=STYLE[index], label=name)
    ttft.set(xlabel='Offered req/s', ylabel='TTFT p95 (s)')
    goodput.set(xlabel='Offered req/s', ylabel='Joint SLO goodput (req/s)')
    for axis in (ttft, goodput):
        axis.grid(alpha=.2)
        axis.legend(frameon=False, fontsize=8)
    fig.suptitle('Natural149 first sequence · independent N1 starts per point')
    fig.savefig(output, dpi=160)
    plt.close(fig)


def render_placement(pairs, output):
    labels = {'TTFT_p95_ms': 'TTFT p95', 'E2E_p95_ms': 'E2E p95',
              'TPOT_p95_ms': 'TPOT p95', 'ITL_p99_ms': 'ITL p99',
              'joint_goodput_req_s': 'Joint goodput'}
    fig, axes = plt.subplots(1, 2, figsize=(9.4, 3.8), sharey=True, layout='constrained')
    for axis, path, title in zip(axes, pairs, ('Cold first sequence', 'Repeated warm sequence'), strict=True):
        values = read(path)
        for row, key in enumerate(labels):
            point = values['ratios'][key]
            center = point['mean_ratio']
            interval = point['paired_start_bootstrap_CI95']
            if center is None:
                continue
            data = [value for value in point['ratio_by_start'] if value is not None]
            axis.scatter(data, [row] * len(data), c=STYLE[0], alpha=.45, s=15)
            axis.scatter(center, row, color=STYLE[0], marker='D', s=33)
            if interval:
                axis.hlines(row, interval[0], interval[1], lw=1.5, color=STYLE[0])
        axis.axvline(1, ls='--', color='#555555', lw=.9)
        axis.set_xscale('log')
        axis.grid(axis='x', alpha=.20)
        axis.set(yticks=range(len(labels)), yticklabels=list(labels.values()),
                 title=f"{title} · {values['independent_pairs']} pairs", xlabel='E VRAM / pinned-host ratio')
    axes[0].invert_yaxis()
    fig.suptitle('Matched encoder-cache placement · independent server starts')
    fig.savefig(output, dpi=160)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--summary', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--timeline', type=Path, nargs=2)
    parser.add_argument('--placement', type=Path, nargs=2, metavar=('COLD', 'WARM'))
    args = parser.parse_args()
    args.out.mkdir(exist_ok=True, parents=True)
    records = [row for run in read(args.summary)['runs'] for row in run['measurements']]
    write_csv(records, args.out / 'online_cases.csv')
    for size, label, title in ((149, 'natural_online', 'Frozen natural-content reuse'),
                                (188, 'working_set_online', 'Controlled cyclic working-set stress'),
                                (91, 'unique_online', 'Zero-image-reuse control')):
        picked = [row for row in records if row['scheduled'] == size and row['failed'] == 0]
        if picked:
            render_metric(picked, args.out / (label + '.png'), title, size)
    render_offered_load(records, args.out / 'natural149_offered_load.png')
    if args.placement:
        render_placement(args.placement, args.out / 'embedding_placement_paired.png')
    if args.timeline:
        render_timeline(*args.timeline, 'measured_1', args.out / 'native_cache_http_timeline.png')


if __name__ == '__main__':
    main()
