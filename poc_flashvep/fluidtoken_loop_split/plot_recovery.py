"""Plot measured recovery data. Never fill absent device activity with idle."""
import argparse
from .common import *


def plot(name):
    configure()
    from ..looped_pd_autonomous_headroom.analysis_lane import quiet_analysis
    with quiet_analysis(name + '_figure'):
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        from ..looped_pd_headroom_recovery.profile_analysis import union
        folder = ROOT / 'raw' / name
        profile = json.loads((folder / 'profile_analysis.json').read_text())
        kernels = json.loads((folder / 'aligned_kernels.json').read_text())
        queue = json.loads((folder / 'submission_queue_audit_timeline.json').read_text())
        config = json.loads((ROOT / 'configs' / (name + '.json')).read_text())
        slo = json.loads((ROOT / 'configs/frozen_SLO.json').read_text())[config['model'] + '_L' + str(config['length'])]
        left, right = kernels['window_start'], kernels['window_end']
        length = right - left
        fig, axes = plt.subplots(4, 1, figsize=(11, 9), sharex=True, constrained_layout=True)
        axes[0].step([row['time'] - left for row in queue],
                     [row['audited_count'] for row in queue], where='post', color='#2456a5')
        axes[0].set_ylabel('D-clear P-ready\nrequests')
        bins = []
        start = left
        while start < right:
            end = min(right, start + .2)
            spans = [(max(start, row['start_monotonic']), min(end, row['end_monotonic']))
                     for row in kernels['kernels'] if row['end_monotonic'] > start and row['start_monotonic'] < end]
            bins.append((start - left, end - start, 100 * union(spans) / (end - start)))
            start = end
        axes[1].bar([row[0] for row in bins], [row[2] for row in bins],
                    width=[row[1] for row in bins], align='edge', color='#348766')
        axes[1].axhline(80, color='black', linestyle='--', label='80% screen')
        axes[1].set_ylim(0, 105)
        axes[1].set_ylabel('P kernel union\n(% per 200 ms)')
        intervals = []
        for line in (folder / 'requests.jsonl').read_text().splitlines():
            row = json.loads(line)
            if row.get('warmup'):
                continue
            for event in row['token_events']:
                if not event['first'] and left <= event['previous_time'] <= event['time'] <= right:
                    intervals.append((event['time'] - left,
                                      1000 * (event['time'] - event['previous_time']) / event['new_tokens']))
        axes[2].scatter([row[0] for row in intervals], [row[1] for row in intervals], s=9, color='#2456a5')
        axes[2].axhline(slo['ITL_ms'], color='#b33c34', linestyle='--', label='Frozen p99 ITL limit')
        axes[2].set_ylabel('D token ITL (ms)\nNot GPU kernel time')
        observations = [json.loads(line) for line in (folder / 'queue_observations.jsonl').read_text().splitlines()]
        credit = []
        for row in observations:
            if left <= row['time'] <= right:
                for value in row['worker_credits'].values():
                    if value['role'] == 'decode':
                        credit.append((row['time'] - left, value['total_blocks'] - value['reserved_blocks']))
        axes[3].step([row[0] for row in credit], [row[1] for row in credit], where='post', color='#348766')
        axes[3].set_ylabel('D free\ncoordinator blocks')
        probes = [json.loads(line) for line in (folder / 'D_probe_requests.jsonl').read_text().splitlines()]
        shown = 0
        for row in probes:
            if left <= row['time'] <= row['finish'] <= right:
                for axis in axes:
                    axis.axvspan(row['time'] - left, row['finish'] - left,
                                 color='#d38b28', alpha=.35)
                shown += 1
        for axis in axes:
            axis.grid(alpha=.2)
            axis.set_xlim(0, length)
        axes[2].legend(loc='upper left')
        axes[3].set_xlabel('Seconds from the same capture start; orange = D-local request lifetime (not GPU duration)')
        status = 'QUALIFIED TIMING' if profile['eligible_for_matched_clean'] else 'REJECTED TIMING: profiler interference exceeds the gate'
        screen = json.loads((folder / 'recovery_joint_screen.json').read_text())
        from .confirm_recovery import operational
        joint = 'SCREEN PASS; independent confirmation required' if operational(screen) else 'JOINT SCREEN FAIL'
        fig.suptitle(f"Huginn R32, L{config['length']}/O{config['cap']}, {config['arrival_rate']} QPS\n"
                     f"{status}; {joint}\n{shown} additional local requests in this interval", fontsize=12)
        output = ROOT / 'figures' / (name + '_same_window')
        fig.savefig(str(output) + '.png', dpi=170)
        fig.savefig(str(output) + '.svg')
        plt.close(fig)
        write(ROOT / 'figures' / (name + '_manifest.json'), dict(
            run=name, input_profile=str(folder / 'profile_analysis.json'),
            png=str(output) + '.png', svg=str(output) + '.svg',
            P_device_activity='MEASURED_CUPTI_KERNEL_INTERVALS',
            D_device_activity='NOT_MEASURED_IN_THIS_P_ONLY_CAPTURE',
            D_ITL='MEASURED_COORDINATOR_TOKEN_EVENTS',
            D_KV='MEASURED_COORDINATOR_CREDIT',
            limitation='D-local requests are not migrated serving requests. This figure does not establish a method speedup.'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('name')
    plot(parser.parse_args().name)
