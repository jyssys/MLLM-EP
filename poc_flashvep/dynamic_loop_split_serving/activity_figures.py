"""Show only the common interval of actual P and D kernel captures."""
from .common import *


def merged_intervals(kernels, left, right):
    intervals = sorted((max(left, row['start_monotonic']), min(right, row['end_monotonic']))
                       for row in kernels if row['end_monotonic'] > left and row['start_monotonic'] < right)
    result = []
    for start, end in intervals:
        if result and start <= result[-1][1]:
            result[-1] = (result[-1][0], max(end, result[-1][1]))
        else:
            result.append((start, end))
    return result


def one(name, plt):
    folder = ROOT / 'raw' / name
    cfg = json.loads((ROOT / 'configs' / (name + '.json')).read_text())
    profiles = [json.loads((folder / filename).read_text())
                for filename in ('profile_analysis.json', 'profile_analysis_decode.json')]
    captures = [json.loads((folder / filename).read_text())
                for filename in ('aligned_kernels.json', 'aligned_kernels_decode.json')]
    window = json.loads((folder / 'window.json').read_text())
    left = max(window['start'] + .2 * (window['last_arrival'] - window['start']),
               *(row['window_start'] for row in captures))
    right = min(window['start'] + .9 * (window['last_arrival'] - window['start']),
                *(row['window_end'] for row in captures))
    assert right > left, 'No shared central P/D capture interval'
    bands = [merged_intervals(row['kernels'], left, right) for row in captures]
    fractions = [sum(end - start for start, end in intervals) / (right - left) for intervals in bands]
    eligible = all(row['eligible_for_matched_clean'] for row in profiles)
    from ..looped_pd_autonomous_headroom.ready_queue_audit import audit
    queue = audit(cfg, dict(observation_start=left, observation_end=right,
        profile_perturbation_qualified=eligible, P_actual_kernel_busy_central=fractions[0],
        P_fraction_reference=None))
    timeline_path = folder / 'submission_queue_audit_timeline.json'
    queue_rows = json.loads(timeline_path.read_text()) if timeline_path.exists() else []
    requests = [json.loads(line) for line in (folder / 'requests.jsonl').read_text().splitlines()
                if not json.loads(line).get('warmup')]
    intervals = [dict(start=event['previous_time'], end=event['time'],
        ms=1000 * (event['time'] - event['previous_time']) / event['new_tokens'],
        delivered_tokens=event['new_tokens']) for row in requests for event in row['token_events']
        if not event['first'] and left <= event['previous_time'] <= event['time'] <= right]
    credit = []
    for line in (folder / 'queue_observations.jsonl').read_text().splitlines():
        row = json.loads(line)
        if left <= row['time'] <= right:
            for peer in row['worker_credits'].values():
                if peer['role'] == 'decode':
                    credit.append(dict(time=row['time'], free=peer['total_blocks'] - peer['reserved_blocks']))
    service_path = folder / 'service_latency_requests.json'
    service = json.loads(service_path.read_text()) if service_path.exists() else []
    moved = [row for row in service if row['D_continuation_to_first_token_ms'] is not None
             and row['first_token'] - row['D_continuation_to_first_token_ms'] / 1000 < right
             and row['first_token'] > left]
    slo = json.loads((ROOT / 'configs/frozen_SLO.json').read_text())[cfg['model'] + '_L' + str(cfg['length'])]
    fig, axes = plt.subplots(4, 1, figsize=(11, 9), sharex=True, constrained_layout=True)
    if queue_rows:
        axes[0].step([row['time'] - left for row in queue_rows],
                     [row['audited_count'] for row in queue_rows], where='post', color='#3977a9')
    else:
        axes[0].text(.03, .5, 'P-ready queue: NOT_MEASURED', transform=axes[0].transAxes)
    axes[0].set_ylabel('D-clear P-ready\nrequests')
    for index, (values, color) in enumerate(zip(bands, ('#3977a9', '#3b8b69'))):
        axes[1].broken_barh([(start - left, end - start) for start, end in values],
                            (index - .3, .6), facecolors=color, rasterized=True)
    axes[1].set_yticks([0, 1], [f'P {100 * fractions[0]:.2f}%', f'D {100 * fractions[1]:.2f}%'])
    axes[1].set_ylabel('Actual kernel union\nBlank = no recorded kernel')
    axes[2].scatter([row['end'] - left for row in intervals], [row['ms'] for row in intervals], s=7, color='#3b8b69')
    axes[2].axhline(slo['ITL_ms'], ls='--', color='#b84037', label='Frozen ITL p99 limit')
    axes[2].set_ylabel('D delivery ITL (ms)\nNot GPU kernel time')
    axes[2].legend(loc='upper right', fontsize=8)
    if credit:
        axes[3].step([row['time'] - left for row in credit], [row['free'] for row in credit],
                     where='post', color='#3b8b69')
    else:
        axes[3].text(.03, .5, 'D coordinator credit: NOT_MEASURED', transform=axes[3].transAxes)
    axes[3].set_ylabel('D free\ncoordinator KV blocks')
    for row in moved:
        start = row['first_token'] - row['D_continuation_to_first_token_ms'] / 1000
        for ax in axes:
            ax.axvspan(max(start, left) - left, min(row['first_token'], right) - left,
                       color='#d68b32', alpha=.18)
    for ax in axes:
        ax.set_xlim(0, right - left)
        ax.grid(alpha=.18)
    axes[-1].set_xlabel('Seconds from the common capture start. Orange = D continuation-to-output enclosure, not exclusive GPU time.')
    status = 'MATCHED CAPTURE SCREEN PASS' if eligible else 'REJECTED TIMING: matched capture screen failed'
    fig.suptitle(f"{cfg['model'].title()} | {cfg['arm']} | L{cfg['length']}/O{cfg['cap']} | {cfg['arrival_rate']} nominal QPS\n"
                 f'{status}. Kernel union is not SM occupancy or physical compute saturation.', fontsize=11)
    target = ROOT / 'figures'
    target.mkdir(exist_ok=True)
    files = []
    for extension in ('png', 'svg'):
        path = target / (name.lower() + '_same_run_activity.' + extension)
        fig.savefig(path, dpi=180)
        files.append(str(path.relative_to(ROOT)))
    plt.close(fig)
    from ..looped_pd_headroom_recovery.gates import percentile
    result = dict(name=name, model=cfg['model'], arm=cfg['arm'],
        status='MEASURED_MATCHED_CAPTURE' if eligible else 'REJECTED_PROFILE_TIMING',
        files=files, common_start=left, common_end=right,
        P_kernel_union=fractions[0], D_kernel_union=fractions[1], queue=queue,
        P_service_proximity='NOT_MEASURED_FOR_THIS_EXECUTION',
        D_ITL_p99_ms=percentile([row['ms'] for row in intervals], .99),
        D_request_TPOT_p95_ms=percentile([row['tpot_ms'] for row in requests
            if any(left <= event['time'] <= right for event in row['token_events'])], .95),
        D_min_free_coordinator_blocks=min((row['free'] for row in credit), default=None),
        D_continuation_enclosures_overlapping_window=len(moved),
        strict_joint_success=False,
        scope='One shared interval in one profiled execution. No physical SM or bandwidth counter is inferred. Device event enclosures are not exclusive kernel time. Kernel-band drawing is rasterized, but metrics use the exact interval union. Subpixel gaps require the raw trace. Request TPOT uses the complete request for requests with output in the interval.',
        sources={str(path.relative_to(ROOT)): digest(path) for path in
                 [folder / 'profile_analysis.json', folder / 'profile_analysis_decode.json',
                  folder / 'aligned_kernels.json', folder / 'aligned_kernels_decode.json',
                  folder / 'requests.jsonl', folder / 'queue_observations.jsonl']})
    write(folder / 'same_run_activity_figure.json', result)
    return result


def main():
    initialize()
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size': 9, 'svg.fonttype': 'none'})
    results = []
    for path in (ROOT / 'raw').glob('k_*/profile_analysis_decode.json'):
        try:
            results.append(one(path.parent.name, plt))
        except (AssertionError, KeyError, FileNotFoundError) as error:
            results.append(dict(name=path.parent.name, status='NOT_MEASURED', reason=repr(error)))
    write(ROOT / 'activity_figure_manifest.json', dict(figures=results,
        scope='Actual shared P/D kernel capture intervals. Missing or rejected evidence cannot certify a motivation gate.'))


if __name__ == '__main__':
    initialize()
    from ..looped_pd_autonomous_headroom.analysis_lane import quiet_analysis
    with quiet_analysis('same_run_kernel_figures'):
        main()
