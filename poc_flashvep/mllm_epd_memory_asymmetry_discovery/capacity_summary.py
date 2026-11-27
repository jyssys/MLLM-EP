"""Compact receipt and figures for the native Main100 residency-only sweep."""

import argparse
import hashlib
import json
from pathlib import Path


GIB = 2**30


def load(file):
    return json.loads(file.read_text())


def sha(file):
    with file.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def summarize(run):
    worker_file = run / 'state/worker_after_full_load.json'
    workers = load(worker_file)
    memory = []
    for role, state in workers.items():
        memory.append({
            'role': role, 'device': state['device_properties'],
            'total_GiB': state['device_total_bytes'] / GIB,
            'free_GiB': state['device_free_bytes'] / GIB,
            'parameters_GiB': state['model_parameter_logical_bytes'] / GIB,
            'vision_GiB': state['model_parameter_bytes_by_prefix'].get('visual', 0) / GIB,
            'language_GiB': state['model_parameter_bytes_by_prefix'].get('language_model', 0) / GIB,
            'kv_GiB': state['unique_KV_storage_bytes'] / GIB,
            'allocated_GiB': state['allocated_bytes'] / GIB,
            'reserved_GiB': state['reserved_bytes'] / GIB,
            'peak_allocated_GiB': state['maximum_allocated_bytes'] / GIB,
        })
    arms = []
    for capacity in (0, 8, 16, 32, 64, 80):
        tag = 'native_strong_free_2p5' if capacity == 0 else f'encoder_arena_{capacity}gib'
        summary_file = run / 'analysis' / f'{tag}_metrics.json'
        metrics = load(summary_file)
        state = load(run / 'state' / f'{tag}_complete.json')
        arena = load(run / 'state' / f'{tag}_ready.json') if capacity else None
        scope_file = run / 'configs' / f'{tag}_arrival.json'
        arms.append({
            'capacity_GiB': capacity, 'scope': 'N1 native EPD with touched residency-only arena',
            'count': metrics['planned_count'], 'completed': metrics['completed_count'],
            'TTFT_p50_ms': metrics['TTFT_p50_ms'], 'TTFT_p95_ms': metrics['TTFT_p95_ms'],
            'TPOT_p95_ms': metrics['TPOT_p95_ms'], 'ITL_p99_ms': metrics['ITL_p99_ms'],
            'joint_SLO': metrics['categories']['JOINT_PASS'],
            'duration_s': state['duration_seconds'],
            'joint_goodput_req_s': metrics['categories']['JOINT_PASS'] / state['duration_seconds'],
            'offered_QPS': load(scope_file)['QPS'], 'arrival_sha256': sha(scope_file),
            'timings_sha256': metrics['raw_SHA256'], 'metrics_sha256': sha(summary_file),
            'arena_free_after_GiB': arena['free_after_bytes'] / GIB if arena else None,
        })
    return {
        'evidence': 'MEASURED_ONLINE', 'weights_and_memory_sha256': sha(worker_file),
        'scope': 'One server start in fixed 0,8,16,32,64,80 order. GPM observer in each arm. '
                 'No authority for latency equivalence or active-cache safety.',
        'normal_working_state': memory, 'residency_arms': arms,
    }


def plot(result, output):
    import matplotlib
    matplotlib.use('Agg')
    matplotlib.rcParams['svg.hashsalt'] = 'mllm-epd-memory-asymmetry'
    import matplotlib.pyplot as plt

    output.mkdir(parents=True, exist_ok=True)
    memory = result['normal_working_state']
    fig, ax = plt.subplots(figsize=(7, 4))
    names = [row['role'].capitalize() for row in memory]
    bottom = [0.0] * len(memory)
    groups = [
        ('Vision weights', [row['vision_GiB'] for row in memory], '#17696b'),
        ('Language weights', [row['language_GiB'] for row in memory], '#429691'),
        ('KV storage', [row['kv_GiB'] for row in memory], '#2d5187'),
        ('Other allocator reservation', [row['reserved_GiB'] - row['parameters_GiB'] - row['kv_GiB'] for row in memory], '#b5c8dd'),
        ('Driver and external allocations', [row['total_GiB'] - row['free_GiB'] - row['reserved_GiB'] for row in memory], '#cbb8a0'),
        ('Free VRAM', [row['free_GiB'] for row in memory], '#e5e7ea'),
    ]
    for label, values, color in groups:
        ax.bar(names, values, bottom=bottom, color=color, label=label)
        bottom = [a + b for a, b in zip(bottom, values, strict=True)]
    for name, height, row in zip(names, bottom, memory, strict=True):
        ax.annotate(f"{row['free_GiB']:.1f} free", (name, height - 10), ha='center')
    ax.set(ylabel='GiB', title='Native EPD memory after Main100')
    ax.legend(loc='upper center', bbox_to_anchor=(.5, -.12), ncol=2, frameon=False)
    fig.tight_layout()
    fig.savefig(output / 'native_memory_breakdown.svg', metadata={'Date': None})
    plt.close(fig)

    arms = result['residency_arms']
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.6))
    capacities = [row['capacity_GiB'] for row in arms]
    axes[0].plot(capacities, [row['TTFT_p95_ms'] / 1000 for row in arms], 'o-', color='#17696b')
    axes[0].set(xlabel='Touched E arena (GiB)', ylabel='TTFT p95 (s)')
    axes[1].plot(capacities, [row['ITL_p99_ms'] for row in arms], 'o-', color='#2d5187', label='ITL p99')
    axes[1].plot(capacities, [row['TPOT_p95_ms'] for row in arms], 's-', color='#925640', label='TPOT p95')
    axes[1].set(xlabel='Touched E arena (GiB)', ylabel='Latency (ms)')
    axes[1].legend(frameon=False)
    fig.suptitle('Main100 at 2.5 req/s · fixed-order residency screen')
    fig.tight_layout()
    fig.savefig(output / 'native_residency_sweep.svg', metadata={'Date': None})
    plt.close(fig)

    for name in ('native_memory_breakdown.svg', 'native_residency_sweep.svg'):
        vector = output / name
        vector.write_text(''.join(line.rstrip() + '\n' for line in vector.read_text().splitlines()))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', required=True, type=Path)
    parser.add_argument('--out', required=True, type=Path)
    parser.add_argument('--figures', type=Path)
    args = parser.parse_args()
    result = summarize(args.run)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + '\n')
    if args.figures:
        plot(result, args.figures)


if __name__ == '__main__':
    main()
