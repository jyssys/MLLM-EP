"""Retain phase-only counts and audit native pending prefill submissions.

Request phase can advance before the submitted GPU work finishes. Host service
spans and pending events are not GPU compute measurements. The original kernel,
service-rate, and SLO screens remain mandatory.
"""
import bisect
import statistics
from .common import *


def audit(cfg, gates):
    if not cfg.get('observe_P_submission_readiness') or 'observation_start' not in gates:
        return None
    activate()
    from ..looped_pd_headroom_recovery.queues import analyze, slope
    folder = ROOT/'raw'/cfg['name']
    left, right = gates['observation_start'], gates['observation_end']
    window = json.loads((folder/'window.json').read_text())
    _, timeline = analyze(cfg['name'], window, observation=(left, right))
    events = []
    for path in folder.glob('P_submission_readiness_*.jsonl'):
        events.extend(json.loads(line) for line in path.read_text().splitlines())
    events.sort(key=lambda r: r['time'])
    clocks = [r['time'] for r in events]
    beginnings = {}
    for path in folder.glob('host_execution_*.json'):
        for event in json.loads(path.read_text()).get('frontiers', []):
            if event['event'] == 'P_FIRST_SUBMISSION':
                beginnings[event['request_id']] = event['time']
    metrics = json.loads((folder/'closed_stats.json').read_text()).get('metrics', [])
    spans = []
    for row in metrics:
        if not row['request_id'].startswith('r0_') or row['transfer_id'] not in beginnings:
            continue
        end = row['timings'].get('prefill_compute_completed')
        if end is not None:
            spans.append((beginnings[row['transfer_id']], row['created_monotonic']+end))
    points = []
    output = []
    for row in timeline:
        t = row['time']
        if not left <= t <= right:
            continue
        index = bisect.bisect_right(clocks, t)-1
        event = events[index] if index >= 0 and t-events[index]['time'] <= .15 else None
        pending = bool(event and any(not ticket['ready'] and 'PREFILL' in ticket['submission_phase'] for ticket in event['tickets']))
        host_service = any(start <= t <= end for start, end in spans)
        admitted = row['admitted_owner_queue_not_inflight_or_connector_blocked']
        if admitted is None:
            continue
        old_slots = row['compute_phase_qualified_P_slot_wait'] or 0
        slots = row['unadmitted_P_slot_wait_D_credit_clear'] if pending or host_service else old_slots
        value = admitted+slots
        points.append((t, value))
        output.append(dict(time=t, phase_only_count=row['ready_backlog_lower_bound'],
                           audited_count=value, pending_prefill_event=pending,
                           native_prefill_host_service=host_service,
                           scope='Host/event evidence qualifies slot ownership only. It does not substitute for GPU kernel evidence.'))
    early = [v for t, v in points if t < (left+right)/2]
    late = [v for t, v in points if t >= (left+right)/2]
    growth = statistics.mean(late)-statistics.mean(early) if early and late else None
    maximum = max((v for _, v in points), default=None)
    queue_pass = growth is not None and growth > 0 and maximum >= 2
    compute_pass = bool(queue_pass and gates.get('profile_perturbation_qualified') and
                        (gates.get('P_actual_kernel_busy_central') or 0) >= .8 and
                        (gates.get('P_fraction_reference') or 0) >= .85)
    result = dict(name=cfg['name'], observation_start=left, observation_end=right,
                  original_phase_only_P_gate=gates.get('P_compute_gate'),
                  audited_P_gate=compute_pass, half_growth=growth, maximum=maximum,
                  slope_per_s=slope(points), samples=len(points),
                  phase_alias_samples=sum(r['audited_count'] != r['phase_only_count'] for r in output),
                  event_samples=len(events), native_host_spans=len(spans),
                  thresholds='Unchanged: positive half growth, maximum >=2, kernel union >=80%, service proximity >=85%, qualified capture.',
                  scope='Submission-ownership measurement correction. The original phase-only result remains available. No physical SM or bandwidth counter is inferred.')
    write(folder/'submission_queue_audit.json', result)
    write(folder/'submission_queue_audit_timeline.json', output)
    return result
