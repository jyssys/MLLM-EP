"""Locate decode delivery gaps without assigning host overlap to GPU compute."""
import argparse
import bisect
import json
import numpy as np
from .common import ROOT,write
from .window_analysis import merge,duration,overlap


def stats(values):
    return dict(n=len(values),mean=float(np.mean(values)),p95=float(np.percentile(values,95)),
        p99=float(np.percentile(values,99)),max=float(np.max(values))) if values else None


def audit(name,reference):
    folder=ROOT/'raw'/name;cfg=json.loads((folder/'configuration.json').read_text())
    ref=json.loads((ROOT/'raw'/reference/'configuration.json').read_text())
    for key in ('dataset_SHA256','arrival_offsets','output_cap','ignore_eos','gpus','token_budget','max_num_seqs'):
        assert cfg[key]==ref[key],('Unmatched contract',key)
    native=json.loads((ROOT/'analysis'/(reference+'_metrics.json')).read_text());limit=1.05*native['ITL_p99_ms']
    records=[json.loads(line) for line in (folder/'requests.jsonl').read_text().splitlines()]
    steps=json.loads((folder/'engine_steps_decode.json').read_text())['rows']
    starts=[s['start'] for s in steps];ends=[s['end'] for s in steps]
    for step in steps:step.update(prefill_requests=set(),decode_requests=set(),offload_prefill=False)
    queries=json.loads((folder/'actual_queries_decode.json').read_text())['rows'];unmapped=0
    for row in queries:
        index=bisect.bisect_right(starts,row['time'])-1
        if index<0 or row['time']>ends[index]:unmapped+=1;continue
        decode=row['start']>=row['original_input_tokens'];step=steps[index]
        step['decode_requests' if decode else 'prefill_requests'].add(row['request_id'])
        if not decode and cfg.get('axis_kind'):step['offload_prefill']=True
    host=[]
    for row in json.loads((folder/'NIXL_host_setup_decode.json').read_text())['rows']:
        if row.get('kind') in ('read_blocks_host','make_transfer_host','submit_transfer_host'):host.append((row['start'],row['end']))
    hp=folder/'hidden_lifetime_decode.json';hidden=[]
    if hp.exists():
        for row in json.loads(hp.read_text())['rows']:
            if row['event']=='D_HIDDEN_READ_DONE':hidden.append((row['time']-row['host_ms']/1000,row['time']))
    rows=[]
    for record in records:
        events=record['token_events'];rid='q'+str(record['sample_id'])
        if events[0]['count']>1:
            rows.extend(dict(request_id=rid,ITL_ms=0.,co_delivered=True) for _ in range(events[0]['count']-1))
        for prior,current in zip(events,events[1:]):
            left,right=prior['worker_time'],current['worker_time']
            selected=steps[max(0,bisect.bisect_right(ends,left)-1):bisect.bisect_right(starts,right)]
            mixed=[s for s in selected if s['offload_prefill'] and s['decode_requests']]
            rows.append(dict(request_id=rid,ITL_ms=1000*(current['time']-prior['time']),
                worker_delivery_gap_ms=1000*(right-left),co_delivered=False,
                overlap_mixed_prefill_decode_host_step=bool(mixed),
                overlap_offload_prefill_host_step=any(s['offload_prefill'] for s in selected),
                overlap_NIXL_host_ms=1000*duration(overlap([(left,right)],host)),
                overlap_hidden_read_host_ms=1000*duration(overlap([(left,right)],hidden)),
                host_step_wall_ms=[1000*(s['end']-s['start']) for s in selected],
                event_count_increment=current['count']-prior['count']))
            rows.extend(dict(request_id=rid,ITL_ms=0.,co_delivered=True)
                for _ in range(current['count']-prior['count']-1))
    assert len(rows)==sum(r['output_tokens']-1 for r in records)
    tail=[r for r in rows if r['ITL_ms']>limit];absolute=[r for r in rows if r['ITL_ms']>34]
    summary=dict(intervals=stats([r['ITL_ms'] for r in rows]),native_ITL_p99_ms=native['ITL_p99_ms'],
        native_5percent_point_limit_ms=limit,point_limit_pass=float(np.percentile([r['ITL_ms'] for r in rows],99))<=limit,
        intervals_above_native_point_limit=len(tail),absolute_ITL_violations=len(absolute),
        tail_with_mixed_host_step=sum(r.get('overlap_mixed_prefill_decode_host_step',False) for r in tail),
        tail_with_offload_prefill_host_step=sum(r.get('overlap_offload_prefill_host_step',False) for r in tail),
        tail_with_NIXL_host=sum(r.get('overlap_NIXL_host_ms',0)>0 for r in tail),
        tail_with_hidden_read=sum(r.get('overlap_hidden_read_host_ms',0)>0 for r in tail),
        measured_engine_steps=len(steps),mixed_steps=sum(s['offload_prefill'] and bool(s['decode_requests']) for s in steps),
        query_to_host_step_unmapped=unmapped)
    value=dict(name=name,reference=reference,summary=summary,rows=rows,evidence='TRACE_ANALYSIS',
        scope='Intervals retain actual client delivery and co-delivered zero intervals. No timestamp changes. '
            'Temporal host overlap is not exclusive GPU time or proof of causality. Async output can lag its GPU query. '
            'The single-start5percent point check is a red-capable diagnostic,not a paired-start CI safety gate. '
            'A controlled admission intervention must test the proposed cause.')
    write(ROOT/'analysis'/(name+'_decode_tail_audit.json'),value)
    print(name,summary,flush=True)
    return value


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--red-fixture',action='store_true');args=parser.parse_args()
    results=[]
    for qps in (1.8,2.8):
        reference=f'C3Static_Discovery200_q{qps}_NativeMatched_v1'
        for method in ('Layer4','Token4','Hybrid4_f0125'):
            results.append(audit(f'C3Static_Discovery200_q{qps}_{method}_v1',reference))
    if args.red_fixture:assert all(r['summary']['point_limit_pass'] for r in results), 'Observed offload arms violate the unchanged Native-relative decode point limit'


if __name__=='__main__':main()
