"""Show measured P/D evidence over one common capture interval."""
import argparse
import math
from .common import *
from .device_queue_audit import merged


def busy_bins(kernels, left, right, width=.25):
    count = math.ceil((right - left) / width)
    seconds = [0.] * count
    spans = merged((max(left, row['start_monotonic']), min(right, row['end_monotonic']))
                   for row in kernels if row['end_monotonic'] > left and row['start_monotonic'] < right)
    for start, end in spans:
        first = max(0, int((start - left) / width))
        last = min(count - 1, int((end - left) / width))
        for index in range(first, last + 1):
            seconds[index] += max(0., min(end, left + (index + 1) * width) - max(start, left + index * width))
    widths = [min(width, right - left - index * width) for index in range(count)]
    return [index * width for index in range(count)], widths, [100 * s / w for s, w in zip(seconds, widths)]


def plot(name):
    configure()
    from ..looped_pd_autonomous_headroom.analysis_lane import quiet_analysis
    with quiet_analysis(name + '_joint_figure'):
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        folder = ROOT / 'raw' / name
        screen = json.loads((folder / 'recovery_joint_screen.json').read_text())
        gate = screen['gates']
        from .static_screen import metrics
        whole = metrics(name)
        left, right = gate['observation_start'], gate['observation_end']
        queue = json.loads((folder / 'device_queue_audit_timeline.json').read_text())
        fig, axes = plt.subplots(5, 1, figsize=(12, 11), sharex=True, constrained_layout=True)
        axes[0].step([r['time'] - left for r in queue],
            [r['admitted_ready'] + r['D_clear_P_slot_wait'] for r in queue],
            where='post', color='#999999', label='D-clear ready/slot wait')
        axes[0].scatter([r['time'] - left for r in queue], [r['qualified_count'] for r in queue],
            s=9, color='#2456a5', label='Conservative device-qualified lower bound')
        axes[0].set_ylabel('P-ready requests\n(queue audit)')
        axes[0].legend(loc='upper left', fontsize=8)
        evidence = {}
        for axis, role, suffix in ((axes[1], 'P', ''), (axes[2], 'D', '_decode')):
            path = folder / ('aligned_kernels' + suffix + '.json')
            if path.exists():
                kernels = json.loads(path.read_text())['kernels']
                capture = json.loads((folder / ('profile_analysis' + suffix + '.json')).read_text())
                measured_left = max(left, capture['window_start_monotonic'])
                measured_right = min(right, capture['window_end_monotonic'])
                if measured_right <= measured_left:
                    axis.text(.5, .5, 'NO COMMON CAPTURE', transform=axis.transAxes, ha='center')
                    evidence[role] = 'NOT_MEASURED_IN_COMMON_INTERVAL'
                    axis.set_ylim(0, 105)
                    axis.set_ylabel(role + ' kernel union\n(% per 250 ms)')
                    continue
                x, widths, busy = busy_bins(kernels, measured_left, measured_right)
                x = [value + measured_left - left for value in x]
                axis.bar(x, busy, width=widths, align='edge', color='#348766')
                for begin, end in ((left, measured_left), (measured_right, right)):
                    if end > begin:
                        axis.axvspan(begin - left, end - left, color='#999999', hatch='//', alpha=.5)
                evidence[role] = dict(kind='MEASURED_SAME_RUN_KERNEL_UNION',
                    capture_intersection=[measured_left, measured_right],
                    uncaptured_seconds=(measured_left - left) + (right - measured_right),
                    scope='Gray edge regions are not captured. They are not measured device idle time.')
            else:
                axis.text(.5, .5, 'NOT_MEASURED', transform=axis.transAxes, ha='center')
                evidence[role] = 'NOT_MEASURED'
            axis.set_ylim(0, 105)
            axis.set_ylabel(role + ' kernel union\n(% per 250 ms)')
        axes[1].axhline(80, color='black', linestyle='--')
        intervals = []
        for line in (folder / 'requests.jsonl').read_text().splitlines():
            row = json.loads(line)
            if row.get('warmup'):
                continue
            for event in row['token_events']:
                if not event['first'] and left <= event['previous_time'] <= event['time'] <= right:
                    intervals.append((event['time'] - left, 1000 * (event['time'] - event['previous_time']) / event['new_tokens']))
        axes[3].scatter([r[0] for r in intervals], [r[1] for r in intervals], s=8)
        axes[3].axhline(gate['frozen_ITL_limit_ms'], color='#b33c34', linestyle='--', label='Frozen p99 ITL limit')
        axes[3].set_ylabel('D token ITL (ms)\nHost observations')
        axes[3].legend(loc='upper left')
        credits = []
        for line in (folder / 'queue_observations.jsonl').read_text().splitlines():
            row = json.loads(line)
            if left <= row['time'] <= right:
                for peer in row['worker_credits'].values():
                    if peer['role'] == 'decode':
                        credits.append((row['time'] - left, peer['total_blocks'] - peer['reserved_blocks']))
        axes[4].step([r[0] for r in credits], [r[1] for r in credits], where='post', color='#348766')
        axes[4].set_ylabel('D free\ncoordinator blocks')
        probes = [json.loads(line) for line in (folder / 'D_probe_requests.jsonl').read_text().splitlines()]
        shown = [r for r in probes if left <= r['time'] <= r['finish'] <= right]
        for probe in shown:
            for axis in axes:
                axis.axvspan(probe['time'] - left, probe['finish'] - left, color='#d38b28', alpha=.3)
        for axis in axes:
            axis.set_xlim(0, right - left)
            axis.grid(alpha=.2)
        axes[4].set_xlabel('Seconds in one common interval. Orange: D-local request lifetime, not exclusive GPU time.')
        interval_status = 'PASS' if gate['D_SLO_gate'] else 'FAIL'
        whole_status = 'PASS' if whole['Decode_SLO_screen'] else 'FAIL'
        fig.suptitle(f"Huginn R32: {name}\n"
            f"Decode SLO: common interval {interval_status}, whole run {whole_status}. "
            f"Exact local probes: {len(shown)}.\nKernel union is not SM occupancy.", fontsize=10)
        output = ROOT / 'figures' / (name + '_joint_device')
        for extension in ('png', 'svg'):
            fig.savefig(str(output) + '.' + extension, dpi=160)
        plt.close(fig)
        write(ROOT / 'figures' / (name + '_joint_device_manifest.json'), dict(
            run=name, interval=[left, right], device_evidence=evidence, completed_local_probes=len(shown),
            common_interval_decode_SLO=gate['D_SLO_gate'], whole_run_decode_SLO=whole['Decode_SLO_screen'],
            png=str(output) + '.png', svg=str(output) + '.svg',
            scope='All panels use the same execution and interval. Local probes are not migrated requests. No method speedup is asserted.'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('name')
    plot(parser.parse_args().name)
