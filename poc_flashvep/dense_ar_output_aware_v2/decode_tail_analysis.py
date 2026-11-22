"""Locate client decode tails in recorded host intervals without GPU attribution."""
import argparse
import bisect
import json
from pathlib import Path
import time
import numpy as np
from .common import ROOT,write
from .c0_analysis import csv_save

def union(values):
    result=[]
    for left,right in sorted(values):
        if right<=left:continue
        if result and left<=result[-1][1]:result[-1][1]=max(result[-1][1],right)
        else:result.append([left,right])
    return result

def clipped_ms(left,right,values):
    return 1000*sum(max(0,min(right,b)-max(left,a)) for a,b in union(values))

def audit(name):
    folder=ROOT/'raw'/name
    requests=[json.loads(s) for s in (folder/'requests.jsonl').read_text().splitlines()]
    steps=json.loads((folder/'engine_steps_decode.json').read_text())['rows']
    assert all(a['end']<=b['start'] for a,b in zip(steps,steps[1:])), 'Host steps must not overlap'
    starts=[s['start'] for s in steps];ends=[s['end'] for s in steps]
    for s in steps:s.update(prefill=set(),decode=set())
    queries=json.loads((folder/'actual_queries_decode.json').read_text())['rows'];unmapped=0
    for q in queries:
        index=bisect.bisect_right(starts,q['time'])-1
        if index<0 or q['time']>ends[index]:unmapped+=1;continue
        steps[index]['decode' if q['start']>=q['original_input_tokens'] else 'prefill'].add(q['request_id'])
    nixl=json.loads((folder/'NIXL_host_setup_decode.json').read_text())['rows']
    nixl=[(r['start'],r['end']) for r in nixl if r.get('kind') in ('read_blocks_host','make_transfer_host','submit_transfer_host')]
    admissions=[json.loads(s) for s in (folder/'admissions.jsonl').read_text().splitlines()]
    admissions=[r for r in admissions if r['role']=='decode']
    cfg=json.loads((folder/'configuration.json').read_text());rows=[]
    for request in requests:
        rid='q'+str(request['sample_id']);events=request['token_events']
        for _ in range(events[0]['count']-1):rows.append(dict(name=name,request_id=rid,prompt_id=request['prompt_id'],ITL_ms=0.,co_delivered=True))
        for previous,current in zip(events,events[1:]):
            left=previous['worker_time'];right=current['worker_time'];assert right>=left
            selected=[s for s in steps[max(0,bisect.bisect_right(ends,left)):bisect.bisect_left(starts,right)]
                if min(right,s['end'])>max(left,s['start'])]
            client=1000*(current['time']-previous['time']);worker=1000*(right-left)
            delivery_delta=1000*((current['time']-current['worker_time'])-(previous['time']-previous['worker_time']))
            assert abs(client-worker-delivery_delta)<1e-4
            engine=clipped_ms(left,right,[(s['start'],s['end']) for s in selected])
            prefill=[(s['start'],s['end']) for s in selected if s['prefill']]
            mixed=[(s['start'],s['end']) for s in selected if s['prefill'] and s['decode']]
            rows.append(dict(name=name,request_id=rid,prompt_id=request['prompt_id'],task=request['task'],
                output_count_before=previous['count'],output_count_after=current['count'],co_delivered=False,
                ITL_ms=client,worker_emission_gap_ms=worker,delivery_delay_delta_ms=delivery_delta,
                current_worker_to_client_ms=1000*(current['time']-current['worker_time']),
                engine_step_host_overlap_ms=engine,not_inside_engine_step_host_ms=max(0,worker-engine),
                prefill_host_step_overlap_ms=clipped_ms(left,right,prefill),
                mixed_prefill_decode_host_step_overlap_ms=clipped_ms(left,right,mixed),
                NIXL_API_host_overlap_ms=clipped_ms(left,right,nixl),
                new_D_registration_count=sum(left<=r['worker_accepted']<right for r in admissions),
                host_steps=len(selected),axis_prefill=bool(cfg.get('axis_kind')),
                evidence='TRACE_ANALYSIS'))
            for _ in range(current['count']-previous['count']-1):rows.append(dict(name=name,request_id=rid,prompt_id=request['prompt_id'],ITL_ms=0.,co_delivered=True))
    metric=json.loads((ROOT/'analysis'/(name+'_metrics.json')).read_text())
    assert len(rows)==metric['ITL_interval_count']==sum(r['output_tokens']-1 for r in requests)
    assert abs(float(np.percentile([r['ITL_ms'] for r in rows],99))-metric['ITL_p99_ms'])<1e-7
    tail=[r for r in rows if r['ITL_ms']>=metric['ITL_p99_ms'] and not r['co_delivered']]
    summary=dict(name=name,intervals=len(rows),ITL_p99_ms=metric['ITL_p99_ms'],max_ITL_ms=metric['max_ITL_ms'],
        every_ITL_SLO_attainment=metric['every_ITL_SLO_attainment'],tail_intervals=len(tail),
        tail_with_prefill_host_step=sum(r['prefill_host_step_overlap_ms']>0 for r in tail),
        tail_with_NIXL_API_host_overlap=sum(r['NIXL_API_host_overlap_ms']>0 for r in tail),
        tail_with_new_registration=sum(r['new_D_registration_count']>0 for r in tail),
        tail_mean_worker_gap_ms=float(np.mean([r['worker_emission_gap_ms'] for r in tail])) if tail else None,
        tail_mean_delivery_delay_delta_ms=float(np.mean([r['delivery_delay_delta_ms'] for r in tail])) if tail else None,
        query_to_host_step_unmapped=unmapped,measured_host_steps=len(steps),
        mixed_host_steps=sum(bool(s['prefill'] and s['decode']) for s in steps),
        scope='Client interval equals worker-emission gap plus delivery-delay change. Co-delivered tokens retain zero client intervals. '
            'Overlapping host categories are not additive causal costs. Engine-step overlap is not exclusive GPU time. '
            'Time outside engine.step is not physical GPU idle. NIXL overlap and registration counts do not prove causality. '
            'The paired independent-start gate remains separate. No safe-window or dynamic-speedup claim follows from this table.')
    write(ROOT/'analysis'/(name+'_decode_tail_host_v2.json'),dict(time=time.time(),summary=summary,rows=rows))
    csv_save(ROOT/'figures'/(name+'_decode_tail_host_v2.csv'),sorted(rows,key=lambda r:-r['ITL_ms'])[:200])
    print('DECODE_TAIL_HOST_AUDIT',summary,flush=True)
    return summary

def plan_analysis(plan_name):
    plan=json.loads((ROOT/'jobs'/plan_name).read_text());summaries=[]
    for block in plan['blocks']:
        for label,name in block['arms'].items():
            if label=='Agg':continue
            receipt=ROOT/'raw'/name/'dispatcher_result.json'
            if not receipt.exists() or not json.loads(receipt.read_text())['complete']:continue
            path=ROOT/'analysis'/(name+'_decode_tail_host_v2.json')
            value=json.loads(path.read_text())['summary'] if path.exists() else audit(name)
            summaries.append(dict(split=block['split'],QPS=block['qps'],start=block['start'],label=label,**value))
    csv_save(ROOT/'figures'/(Path(plan_name).stem+'_decode_tail_host_summary_v2.csv'),summaries)
    return summaries

if __name__=='__main__':
    from pathlib import Path
    p=argparse.ArgumentParser();p.add_argument('--name');p.add_argument('--plan');a=p.parse_args()
    if a.name:audit(a.name)
    else:plan_analysis(a.plan)
