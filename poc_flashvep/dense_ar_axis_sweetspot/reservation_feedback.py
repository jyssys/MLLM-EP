"""Separate mandatory reservation RTT from source Forward service."""
import collections
import json
import statistics
import time
from .common import ROOT,write
from .historical_audit import records


def analyze(name):
    folder=ROOT/'raw'/name
    cfg=json.loads((ROOT/'configs'/(name+'.json')).read_text())
    assert cfg['P_group_size']==1
    decisions=records(folder/'online_decisions.jsonl')
    reservations={r['generation']:r for r in records(folder/'reservation_observations.jsonl')}
    host=json.loads((ROOT/'analysis'/(name+'_host_paths.json')).read_text())
    paths=sorted(host['request_paths'],key=lambda r:int(r['request_id'][1:]))
    requests={'q'+str(r['sample_id']):r for r in records(folder/'requests.jsonl')}
    assert len(paths)==len(decisions)==len(reservations)==cfg['count']
    assert all(r['accepted'] for r in reservations.values())
    rows=[]
    for index,(path,decision) in enumerate(zip(paths,decisions)):
        assert path['request_id']=='q'+str(index) and decision['observation']['group_size']==1
        reservation=reservations[decision['generation']]
        response=reservation['source_receive'];neg=reservation['negotiation_ms']/1000
        begin=response-neg;submit=path['boundaries_monotonic'][1]
        assert begin<=response<=submit+1e-7
        active=decision['active_D_requests'];counts=decision['active_delivered_counts']
        phase=('HOST_IDLE_NO_ACTIVE_COHORT' if not active else
            'HOST_COHORT_PREPARATION' if min(counts.get(r,0) for r in active)==0 else 'HOST_COHORT_DECODE')
        item=dict(request_id=path['request_id'],generation=decision['generation'],
            reservation_RTT_ms=1000*neg,D_host_phase_at_decision=phase,
            post_reservation_to_P_send_ms=1000*(submit-response),
            source_submit_to_ready_ms=path['spans_ms']['P_submit_to_source_notification'])
        if index:
            prior_ready=paths[index-1]['boundaries_monotonic'][2]
            outstanding=sum(p['boundaries_monotonic'][2]<=prior_ready and
                requests[p['request_id']]['finish']>prior_ready for p in paths)
            gap=submit-prior_ready
            parts=dict(before_reservation_ms=1000*(begin-prior_ready),
                reservation_RTT_ms=1000*neg,post_reservation_ms=1000*(submit-response))
            assert abs(sum(parts.values())-1000*gap)<1e-5
            item.update(P_inter_request_gap_ms=1000*gap,gap_parts_ms=parts,
                already_arrived_at_prior_ready=path['boundaries_monotonic'][0]<=prior_ready,
                source_outstanding_at_prior_ready=outstanding,
                source_window_blocked_at_prior_ready=outstanding>=cfg['source_outstanding_limit'])
        rows.append(item)
    backlogged=[r for r in rows if r.get('already_arrived_at_prior_ready')]
    eligible=[r for r in backlogged if not r['source_window_blocked_at_prior_ready']]
    phase=collections.defaultdict(list)
    for r in rows:phase[r['D_host_phase_at_decision']].append(r['reservation_RTT_ms'])
    value=dict(name=name,method=cfg['evaluation_arm'],layers=cfg['static_layers'],output=cfg['cap'],
        batch=cfg['D_admission_batch'],stage=cfg['matrix_stage'],seed=cfg['arrival_seed'],QPS=cfg['arrival_rate'],
        requests=rows,eligible_backlogged_nonwindow_samples=len(eligible),
        mean_backlogged_nonwindow_P_gap_ms=statistics.mean(r['P_inter_request_gap_ms'] for r in eligible) if eligible else None,
        median_below50ms_P_gap_ms=statistics.median(r['P_inter_request_gap_ms'] for r in eligible
            if 0<=r['P_inter_request_gap_ms']<50) if any(0<=r['P_inter_request_gap_ms']<50 for r in eligible) else None,
        mean_backlogged_nonwindow_gap_parts_ms={k:statistics.mean(r['gap_parts_ms'][k] for r in eligible)
            for k in ('before_reservation_ms','reservation_RTT_ms','post_reservation_ms')} if eligible else {},
        reservation_RTT_by_host_phase={p:dict(samples=len(v),mean_ms=statistics.mean(v),maximum_ms=max(v)) for p,v in phase.items()},
        evidence='TRACE_ANALYSIS',
        scope='Each P prefix is submitted only after its destination reservation ACK. Gap components are exact host-clock intervals. Worker control messages are handled before engine.step, not by a separate control thread. Credit rejection0 does not imply zero reservation latency. Logical D phase is not a GPU kernel trace. Retrospective window classification uses completed timestamps only for analysis, never online prediction.')
    write(ROOT/'analysis'/(name+'_reservation_feedback.json'),value)
    return value


def main():
    data=json.loads((ROOT/'analysis/independent_start_statistics.json').read_text())
    values=[analyze(r['name']) for r in data['starts']]
    write(ROOT/'analysis/reservation_feedback_index.json',dict(time=time.time(),runs=values,
        evidence='TRACE_ANALYSIS',scope='Fresh clean fixed-axis traces. The reservation RTT is on the mandatory P admission path. Host-step feedback can alter effective P service without changing GEMM geometry.'))
    print('RESERVATION_FEEDBACK_TRACES',len(values),flush=True)


if __name__=='__main__':
    main()
