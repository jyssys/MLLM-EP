"""Plot measured observations from one run and one shared capture window."""
import argparse
import csv
from .common import *
from .analysis_lane import quiet_analysis
from .aligned_gates import one
from .ready_queue_audit import audit


def lines(path):
    return [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []


def render(name):
    activate()
    from ..looped_pd_headroom_recovery.profile_analysis import union
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    folder = ROOT/'raw'/name
    assert not (folder/'supervisor_recovery.json').exists(), 'Unsupervised run is excluded'
    cfg = json.loads((ROOT/'configs'/(name+'.json')).read_text())
    profile = json.loads((folder/'profile_analysis.json').read_text())
    assert profile['status'] == 'PARSED'
    gates = one(cfg)
    checked = audit(cfg, gates)
    left, right = gates['observation_start'], gates['observation_end']
    kernels = json.loads((folder/'aligned_kernels.json').read_text())['kernels']
    bins = []
    step = min(1., (right-left)/20)
    cursor = left
    while cursor < right:
        end = min(cursor+step, right)
        spans = [(max(cursor, r['start_monotonic']), min(end, r['end_monotonic']))
                 for r in kernels if r['end_monotonic'] > cursor and r['start_monotonic'] < end]
        bins.append(dict(start_s=cursor-left, end_s=end-left,
                         P_kernel_union_percent=100*union(spans)/(end-cursor)))
        cursor = end
    queue_path = folder/'submission_queue_audit_timeline.json'
    queue = json.loads(queue_path.read_text()) if queue_path.exists() else []
    credits = [(r['time']-left, p['total_blocks']-p['reserved_blocks'])
               for r in lines(folder/'queue_observations.jsonl') if left <= r['time'] <= right
               for p in r['worker_credits'].values() if p['role'] == 'decode']
    itls = [(e['time']-left, 1000*(e['time']-e['previous_time'])/e['new_tokens'])
            for r in lines(folder/'requests.jsonl') if not r.get('warmup')
            for e in r['token_events'] if not e['first'] and left <= e['previous_time'] <= e['time'] <= right]
    probes = [r for r in lines(folder/'D_probe_requests.jsonl') if r['finish'] >= left and r['time'] <= right]
    fig, axes = plt.subplots(4, 1, figsize=(11, 10), sharex=True, constrained_layout=True)
    axes[0].stairs([b['P_kernel_union_percent'] for b in bins],
                   [b['start_s'] for b in bins]+[bins[-1]['end_s']], color='#1769aa', fill=True, alpha=.55)
    axes[0].axhline(80, color='black', ls='--', lw=1)
    axes[0].set(ylabel='P kernel union (%)', ylim=(0, 105))
    axes[0].text(.01, .05, 'Device events. Not SM occupancy.', transform=axes[0].transAxes, fontsize=9)
    axes[1].step([r['time']-left for r in queue], [r['audited_count'] for r in queue], where='post', color='#914912')
    axes[1].set(ylabel='P-ready pressure\n(requests)', ylim=(0, None))
    secondary = axes[1].twinx()
    secondary.plot([r[0] for r in credits], [r[1] for r in credits], color='#25743c', alpha=.7)
    secondary.set_ylabel('D free reservation blocks', color='#25743c')
    axes[2].scatter([r[0] for r in itls], [r[1] for r in itls], s=8, alpha=.5)
    axes[2].axhline(gates['frozen_ITL_limit_ms'], color='#a82828', ls='--', label='Frozen ITL p99 limit')
    axes[2].set(ylabel='Observed ITL (ms)', ylim=(0, None))
    axes[2].legend(fontsize=8)
    for i, row in enumerate(probes):
        start, end = max(left, row['time']), min(right, row['finish'])
        axes[3].broken_barh([(start-left, end-start)], (i-.3, .6), facecolors='#3b8067')
        axes[3].scatter([(start+end)/2-left], [i], s=14, color='#3b8067')
    axes[3].set(ylabel='D-local Prefill probe', xlabel='Seconds from shared observation start',
                xlim=(0, right-left), yticks=list(range(0, len(probes), max(1, len(probes)//8))))
    axes[3].text(.01, .93, 'Host request spans. These bars do not show GPU busy or GPU idle.',
                 transform=axes[3].transAxes, fontsize=9, va='top')
    qualified = profile['eligible_for_matched_clean']
    starts = [r for r in lines(folder/'capture_events.jsonl') if r.get('event') == 'START']
    start_pause = starts[0].get('profiler_start_call_seconds') if starts else None
    pause_note = f'\nProfiler start host call: {start_pause:.3f}s before this capture. Local disturbance must be assessed.' if start_pause is not None and start_pause > .1 else ''
    fig.suptitle(f"{cfg['model'].capitalize()} {cfg['length']}/{cfg['cap']} · {cfg['arrival_rate']} req/s · one run\n"
        f"{'Qualified capture' if qualified else 'PERTURBED DIAGNOSTIC — NOT CLEAN PROOF'}"
        f" · wall difference {100*profile['matched_wall_difference_fraction']:+.2f}%"+pause_note, fontsize=12)
    target = ROOT/'figures'/(name+'_same_window.png')
    fig.savefig(target, dpi=180)
    fig.savefig(target.with_suffix('.svg'))
    plt.close(fig)
    with (folder/'same_window_kernel_bins.csv').open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(bins[0]))
        writer.writeheader()
        writer.writerows(bins)
    measured_captures = [r for p in folder.glob('graph_captures_*.jsonl') for r in lines(p) if r.get('measured_phase')]
    write(folder/'same_window_figure_manifest.json', dict(name=name, figure=str(target.relative_to(ROOT)),
        vector_figure=str(target.with_suffix('.svg').relative_to(ROOT)), measured_phase_captures=measured_captures,
        profiler_start_host_seconds=start_pause,
        gates=gates, queue_audit=checked, evidence='MEASURED_AND_TRACE_ANALYSIS',
        scope='All panels use this run and the same interval. D kernel occupancy is NOT_MEASURED. Host probe spans are not GPU timelines.'))
    return str(target)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('names', nargs='+')
    args = parser.parse_args()
    with quiet_analysis('same_window_figures'):
        for name in args.names:
            print(render(name), flush=True)
