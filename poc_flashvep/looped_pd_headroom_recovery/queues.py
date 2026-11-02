"""Exclude D-blocked admissions from the compute-ready backlog screen."""
import statistics
from .common import ROOT
from .analyze import records,table


def slope(points):
    if len(points)<2:return None
    mx=statistics.mean(t for t,v in points)
    my=statistics.mean(v for t,v in points)
    denominator=sum((t-mx)**2 for t,v in points)
    return sum((t-mx)*(v-my) for t,v in points)/denominator if denominator else None


def analyze(name,window,observation=None):
    import json
    folder=ROOT/'raw'/name
    left=window['start']+.2*(window['last_arrival']-window['start'])
    right=window['start']+.9*(window['last_arrival']-window['start'])
    if observation is not None:left,right=observation
    local=[]
    for path in folder.glob('memory_progress_*.jsonl'):
        local.extend(r for r in records(path) if r['role']=='prefill' and 'all_submitted_unretired_ids' in r)
    local.sort(key=lambda r:r['time'])
    timeline=[]
    for row in records(folder/'admission_reasons.jsonl'):
        Dblocked=sum(any(r.startswith('DECODE_') for r in q['reasons']) for q in row['waiting'])
        # No D KV/slot refusal and no P KV refusal. The remaining P slot
        # wait is reported separately from already admitted P-ready work.
        unadmitted=sum('PREFILL_NO_SLOT' in q['reasons'] and
            all(r=='PREFILL_NO_SLOT' for r in q['reasons']) for q in row['waiting'])
        earlier=[v for v in local if v['time']<=row['time'] and row['time']-v['time']<=1.]
        admitted=len(earlier[-1]['queued_not_inflight_not_connector_blocked_ids']) if earlier else None
        compute_count=sum(not any(x in request['phase'].lower() for x in ('wait','receiv','transfer','finish'))
            for request in earlier[-1]['requests']) if earlier else None
        qualified_slots=unadmitted if compute_count else 0
        timeline.append(dict(name=name,time=row['time'],central_window=left<=row['time']<=right,
            coordinator_D_credit_blocked=Dblocked,
            unadmitted_P_slot_wait_D_credit_clear=unadmitted,
            admitted_owner_queue_not_inflight_or_connector_blocked=admitted,
            P_owner_compute_phase_requests=compute_count,
            compute_phase_qualified_P_slot_wait=qualified_slots if admitted is not None else None,
            ready_backlog_lower_bound=qualified_slots+admitted if admitted is not None else None,owner_sample_available=admitted is not None,
            scope='Distinct admitted and unadmitted populations. D-blocked waiting is excluded. Every submitted unretired owner row is excluded, even if its event completed. P-slot pressure joins the screen only with an observed compute-phase owner. Old snapshots without this contract remain unavailable. This is not by itself GPU saturation proof.'))
    points=[(r['time'],r['ready_backlog_lower_bound']) for r in timeline if r['central_window'] and r['ready_backlog_lower_bound'] is not None]
    mid=(left+right)/2
    a=[v for t,v in points if t<mid]
    b=[v for t,v in points if t>=mid]
    result=dict(P_ready_lower_bound_slope_per_s=slope(points),
        P_ready_lower_bound_half_growth=statistics.mean(b)-statistics.mean(a) if a and b else None,
        P_ready_lower_bound_max=max((v for t,v in points),default=None),
        P_owner_queue_available=bool(local),
        queue_scope='Verified owner queue plus compute-phase-qualified D-clear P-slot pressure. Old snapshots cannot prove the unretired-ticket exclusion and are not backfilled. GPU compute, capacity and non-dominant transfer/host gates remain mandatory.')
    return result,timeline


def main():
    import json
    rows=[];timeline=[]
    for path in (ROOT/'raw').glob('*/window.json'):
        value,series=analyze(path.parent.name,json.loads(path.read_text()))
        rows.append(dict(name=path.parent.name,**value))
        timeline.extend(series)
    table('p_ready_queue_analysis.csv',rows)
    table('p_ready_queue_timeline.csv',timeline)
    return rows


if __name__=='__main__':main()
