"""Qualify P-slot queue samples with actual same-run GPU kernel intervals.

Synchronous submission can finish before the scheduler samples its ticket.
This audit uses measured device timestamps, not a host-service-span fallback.
"""
import bisect
import statistics
from .common import *


def merged(intervals):
    result = []
    for left, right in sorted(intervals):
        if result and left <= result[-1][1]:
            result[-1][1] = max(result[-1][1], right)
        else:
            result.append([left, right])
    return result


def check(name):
    configure()
    from ..looped_pd_headroom_recovery.queues import analyze, slope
    folder = ROOT / 'raw' / name
    screen = json.loads((folder / 'recovery_joint_screen.json').read_text())
    gates = screen['gates']
    left, right = gates['observation_start'], gates['observation_end']
    profile = json.loads((folder / 'profile_analysis.json').read_text())
    margin = profile['anchor_uncertainty_s'] + profile['clock_mapping_error_s']
    device = json.loads((folder / 'aligned_kernels.json').read_text())
    spans = merged((r['start_monotonic'], r['end_monotonic']) for r in device['kernels'])
    starts = [r[0] for r in spans]
    _, timeline = analyze(name, json.loads((folder / 'window.json').read_text()), observation=(left, right))
    points, details = [], []
    for row in timeline:
        t = row['time']
        if not left <= t <= right:
            continue
        admitted = row['admitted_owner_queue_not_inflight_or_connector_blocked']
        if admitted is None:
            continue
        index = bisect.bisect_right(starts, t) - 1
        active = index >= 0 and spans[index][0] + margin <= t <= spans[index][1] - margin
        count = admitted + (row['unadmitted_P_slot_wait_D_credit_clear'] if active else 0)
        points.append((t, count))
        details.append(dict(time=t, P_kernel_active=active, admitted_ready=admitted,
            D_clear_P_slot_wait=row['unadmitted_P_slot_wait_D_credit_clear'], qualified_count=count))
    early = [v for t, v in points if t < (left + right) / 2]
    late = [v for t, v in points if t >= (left + right) / 2]
    growth = statistics.mean(late) - statistics.mean(early) if early and late else None
    maximum = max((v for _, v in points), default=None)
    passed = growth is not None and growth > 0 and maximum >= 2
    result = dict(name=name, observation_start=left, observation_end=right,
        half_growth=growth, maximum=maximum, slope_per_s=slope(points), samples=len(points),
        kernel_qualified_samples=sum(r['P_kernel_active'] for r in details),
        timestamp_uncertainty_margin_s=margin, queue_screen_pass=passed,
        scope='Measured P kernel timestamps qualify D-credit-clear P-slot waiting. No host-span fallback. No SM-utilization inference.',
        unchanged_thresholds='Positive half-window growth and maximum at least two. Independent GPU union, service, SLO, and perturbation gates remain required.')
    write(folder / 'device_queue_audit.json', result)
    write(folder / 'device_queue_audit_timeline.json', details)
    return result
