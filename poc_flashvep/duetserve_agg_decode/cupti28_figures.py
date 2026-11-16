"""Plot same-run kernel unions without promoting a failed clean gate."""
import json

from .analyze import records
from .common import ROOT
from .kernel_trace import gaps


def render(plt):
    path = ROOT / 'analysis/api28_Native_serving_gate.json'
    if not path.exists():
        return
    result = json.loads(path.read_text())
    if not result.get('role_windows'):
        return
    folder = ROOT / 'raw' / result['names']['CUPTI']
    measured = json.loads((folder / 'measurement_begin.json').read_text())['time']
    begin_host = measured + 10.1
    converted = {}
    for role, window in result['role_windows'].items():
        a = json.loads((folder / f'activity_{role}_begin.json').read_text())['anchor']
        b = json.loads((folder / f'activity_{role}_end.json').read_text())['anchor']
        scale = (b['cupti_ns'] - a['cupti_ns']) / (b['host_midpoint'] - a['host_midpoint'])
        left = a['cupti_ns'] + (begin_host - a['host_midpoint']) * scale
        idle = [[(x - left) / scale, (y - left) / scale] for x, y in window['idle_windows_CUPTI_ns']]
        converted[role] = dict(idle=idle, busy=gaps(idle, 0, 19.8))
    largest = max(converted['decode']['idle'], key=lambda row: row[1] - row[0])
    origin = max(0, min(largest[0] - .08, 19.0))
    finish = min(19.8, origin + .8)
    fig, axes = plt.subplots(4, 1, figsize=(11.5, 7.4), sharex=True)
    for ax, role, label in zip(axes[:2], ('prefill', 'decode'), ('P / GPU0', 'D / GPU1')):
        busy = [((max(x, origin) - origin) * 1000, (min(y, finish) - max(x, origin)) * 1000)
                for x, y in converted[role]['busy'] if y > origin and x < finish]
        ax.broken_barh(busy, (.2, .6), facecolors='#009e73')
        ax.set(ylim=(0, 1), yticks=[.5], yticklabels=[label])
    ready = records(folder / 'native_handoffs.jsonl')
    times = [(r['time'] - begin_host - origin) * 1000 for r in ready
             if begin_host + origin <= r['time'] <= begin_host + finish]
    axes[2].plot(times, [.5] * len(times), '|', color='#d55e00', markersize=15)
    axes[2].set(ylim=(0, 1), yticks=[.5], yticklabels=['Host P-ready\nnotification'])
    count = 0
    for row in records(folder / 'requests.jsonl'):
        times = [(v['worker_time'] - begin_host - origin) * 1000 for v in row['token_events']
                 if begin_host + origin <= v['worker_time'] <= begin_host + finish]
        if times and count < 4:
            axes[3].plot(times, [count] * len(times), '.-', markersize=3, label=f"D_req{row['sample_id']}")
            count += 1
    axes[3].legend(fontsize=7, loc='upper right')
    axes[3].set(xlabel='Time within the same serving run (ms)', ylabel='Worker token\nemission')
    for ax in axes:
        ax.set_xlim(0, (finish - origin) * 1000)
        ax.grid(alpha=.2)
    status = 'Qualified short window' if result['qualified'] else 'Diagnostic only: clean gate NOT PASSED'
    fig.suptitle('Native PD / CUDA12.9 CUPTI / ' + status + '\nGreen: actual kernel union. Blank: kernel inactivity, not a decode-safe budget.', fontsize=10)
    fig.tight_layout()
    for suffix in ('.png', '.pdf'):
        fig.savefig(ROOT / 'figures' / ('api28_native_same_run_pd_kernel_timeline' + suffix), dpi=180, bbox_inches='tight')
    plt.close(fig)
    import numpy as np
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.7))
    for ax, role in zip(axes, ('prefill', 'decode')):
        durations = [(y - x) * 1000 for x, y in converted[role]['idle'] if y > x]
        bins = np.geomspace(min(durations), max(durations), 60)
        ax.hist(durations, bins=bins, color='#0072b2', alpha=.8)
        ax.set(xscale='log', yscale='log', xlabel='Continuous kernel-inactivity duration (ms)',
               ylabel='Window count', title=('P / GPU0' if role == 'prefill' else 'D / GPU1'))
        ax.grid(alpha=.2)
    fig.suptitle('Native PD / same19.8-second interval / ' + status, fontsize=11)
    fig.tight_layout()
    for suffix in ('.png', '.pdf'):
        fig.savefig(ROOT / 'figures' / ('api28_native_kernel_idle_window_histogram' + suffix), dpi=180, bbox_inches='tight')
    plt.close(fig)
