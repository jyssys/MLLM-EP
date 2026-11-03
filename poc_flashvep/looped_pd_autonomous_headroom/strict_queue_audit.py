"""Audit queue growth without the native host-service fallback."""
import argparse
import statistics
from .common import *
from .analysis_lane import quiet_analysis


def check(name):
    activate()
    from ..looped_pd_headroom_recovery.queues import analyze, slope
    folder = ROOT/'raw'/name
    previous = json.loads((folder/'submission_queue_audit.json').read_text())
    left, right = previous['observation_start'], previous['observation_end']
    _, timeline = analyze(name, json.loads((folder/'window.json').read_text()), observation=(left, right))
    readiness = {r['time']: r for r in json.loads((folder/'submission_queue_audit_timeline.json').read_text())}
    points = []
    for r in timeline:
        event = readiness.get(r['time'])
        admitted = r['admitted_owner_queue_not_inflight_or_connector_blocked']
        if event is None or admitted is None:
            continue
        # An unretired Prefill GPU event must qualify the unadmitted P-slot wait.
        # A host service span alone does not qualify that wait in this audit.
        pending = event['pending_prefill_event']
        count = admitted+(r['unadmitted_P_slot_wait_D_credit_clear'] if pending else 0)
        points.append((r['time'], count))
    early = [v for t, v in points if t < (left+right)/2]
    late = [v for t, v in points if t >= (left+right)/2]
    growth = statistics.mean(late)-statistics.mean(early) if early and late else None
    maximum = max((v for _, v in points), default=None)
    result = dict(name=name, observation_start=left, observation_end=right,
        half_growth=growth, maximum=maximum, slope_per_s=slope(points), samples=len(points),
        queue_screen_pass=growth is not None and growth > 0 and maximum >= 2,
        original_audited_queue_half_growth=previous['half_growth'],
        scope='Sensitivity audit. Only fresh pending Prefill-event evidence qualifies P-slot waiting. No host-span fallback. Not a GPU compute measurement.')
    write(folder/'event_only_queue_audit.json', result)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('names', nargs='+')
    args = parser.parse_args()
    with quiet_analysis('event_only_queue_sensitivity'):
        for name in args.names:
            print(json.dumps(check(name)), flush=True)
