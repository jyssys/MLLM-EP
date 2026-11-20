"""Measure observed partial-continuation boundaries, not exclusive GPU cost."""
import json
from .common import ROOT,write
from .qualification import records
from .serving_analysis import stats


def contiguous_queries(queries,start,end):
    cursor=start
    for row in queries:
        count=max(0,min(row['count'],end-row['start']))
        if count==0:continue
        if row['start']!=cursor:return False
        cursor+=count
    return cursor==end


def analyze(name):
    folder=ROOT/'raw'/name;cfg=json.loads((folder/'configuration.json').read_text())
    outputs=records(folder/'requests.jsonl');handoffs={r['request_id']:r for r in records(folder/'handoffs.jsonl')}
    queries={role:json.loads((folder/f'actual_queries_{role}.json').read_text())['rows'] for role in ('prefill','decode')}
    admissions=records(folder/'admissions.jsonl');decisions={};decision_events={}
    if (folder/'axis_decisions.jsonl').exists():
        decision_events={r['request_id']:r for r in records(folder/'axis_decisions.jsonl')}
        decisions={rid:r['action'] for rid,r in decision_events.items()}
    rows=[]
    for output in outputs:
        rid=f'q{output["sample_id"]}';length=output['input_tokens'];action=decisions.get(rid,dict(kind=cfg.get('axis_kind','Native'),cut=cfg.get('axis_cut',0)))
        source=[q for q in queries['prefill'] if q['request_id']==rid];dest=[q for q in queries['decode'] if q['request_id']==rid]
        first_P=min(q['time'] for q in source);first_D=min(q['time'] for q in dest);ready=handoffs[rid]['source_ready']
        source_count=sum(q['count'] for q in source);dest_prompt=sum(max(0,min(q['count'],length-q['start'])) for q in dest)
        source_cursor=0;source_coverage=True
        for query in source:source_coverage &= query['start']==source_cursor;source_cursor+=query['count']
        dstarts=[r for r in admissions if r['role']=='decode' and r['request_id']==rid]
        pstarts=[r for r in admissions if r['role']=='prefill' and r['request_id']==rid]
        P_dispatch=decision_events[rid]['time'] if rid in decision_events else None
        first_worker=output['token_events'][0]['worker_time']
        cut=action['cut'];axis=action['kind'];P_rows=source_count*((36-cut) if axis=='Layer' else 36)
        D_rows=dest_prompt*(cut if axis=='Layer' else 36)
        expected_extra=36 if axis=='Native' else 0
        expected_P_end=source_count if axis=='Token' else length
        expected_D_start=source_count if axis=='Token' else length-1 if axis=='Native' else 0
        source_exact=contiguous_queries(source,0,expected_P_end)
        destination_exact=contiguous_queries(dest,expected_D_start,length)
        singleton=all(q['requests']==1 for q in source+dest)
        P_padding=sum(q['padded_queries']-q['actual_batch_queries'] for q in source) if singleton else None
        D_padding=sum(q['padded_queries']-q['actual_batch_queries'] for q in dest if q['start']<length) if singleton else None
        rows.append(dict(request_id=rid,prompt_id=output['prompt_id'],task=output['task'],input_tokens=length,
            action=action,P_actual_query_rows=source_count,D_actual_prompt_queries=dest_prompt,
            P_token_layer_rows=P_rows,D_token_layer_rows=D_rows,
            P_padding_query_rows=P_padding,D_prompt_padding_query_rows=D_padding,
            P_padding_token_layer_rows=P_padding*((36-cut) if axis=='Layer' else 36) if singleton else None,
            D_prompt_padding_token_layer_rows=D_padding*(cut if axis=='Layer' else 36) if singleton else None,
            source_query_coverage_exact=source_exact,destination_query_coverage_exact=destination_exact,
            no_missing_duplicate_prompt_work=bool(source_coverage and source_exact and destination_exact and P_rows+D_rows==36*length+expected_extra),
            P_queue_ms=1000*(first_P-output['arrival']),P_prepare_to_ready_ms=1000*(ready-first_P),
            P_coordinator_wait_ms=1000*(P_dispatch-output['arrival']) if P_dispatch is not None else None,
            P_dispatch_to_worker_accept_ms=1000*(pstarts[0]['worker_accepted']-P_dispatch) if P_dispatch is not None and pstarts else None,
            P_worker_accept_to_prepare_ms=1000*(first_P-pstarts[0]['worker_accepted']) if pstarts else None,
            ready_to_D_accept_ms=1000*(dstarts[0]['worker_accepted']-ready) if dstarts else None,
            D_accept_to_prepare_ms=1000*(first_D-dstarts[0]['worker_accepted']) if dstarts else None,
            ready_to_D_prepare_ms=1000*(first_D-ready),D_prepare_to_first_worker_output_ms=1000*(output['token_events'][0]['worker_time']-first_D),
            ready_to_first_worker_output_ms=1000*(output['token_events'][0]['worker_time']-ready),
            worker_first_to_last_output_ms=1000*(output['token_events'][-1]['worker_time']-output['token_events'][0]['worker_time']),
            D_generation_query_rows=sum(q['count'] for q in dest if q['start']>=length),
            emitted_tokens_after_first=output['output_tokens']-1,
            D_generation_rows_minus_emitted_after_first=sum(q['count'] for q in dest if q['start']>=length)-(output['output_tokens']-1),
            output_tokens=output['output_tokens'],TTFT_ms=output['TTFT_ms']))
        rows[-1]['first_worker_to_client_ms']=1000*(output['first_token']-first_worker)
        pieces=(first_P-output['arrival'],ready-first_P,first_D-ready,first_worker-first_D,output['first_token']-first_worker)
        rows[-1]['TTFT_partition_residual_ms']=output['TTFT_ms']-1000*sum(pieces)
        assert abs(rows[-1]['TTFT_partition_residual_ms'])<1e-3
    fields=('P_queue_ms','P_coordinator_wait_ms','P_dispatch_to_worker_accept_ms','P_worker_accept_to_prepare_ms',
            'P_prepare_to_ready_ms','ready_to_D_accept_ms','D_accept_to_prepare_ms','ready_to_D_prepare_ms',
            'D_prepare_to_first_worker_output_ms','ready_to_first_worker_output_ms','first_worker_to_client_ms','worker_first_to_last_output_ms')
    result=dict(name=name,rows=rows,summary={key:stats(r[key] for r in rows if r[key] is not None) for key in fields},
        all_prompt_work_accounted=all(r['no_missing_duplicate_prompt_work'] for r in rows),
        isolated_destination_service=bool(cfg.get('serial')),evidence='TRACE_ANALYSIS',
        scope='Observed host boundaries. Serial destination service excludes other request queues but includes transport, '
        'submission, model, sampling and delivery. Async EOS lookahead can execute an un-emitted query. '
        'Client completion is not a device-wide drain boundary. None of these clocks is exclusive GPU kernel time.')
    write(ROOT/'analysis'/(name+'_axis_timeline.json'),result);return result


if __name__=='__main__':
    import sys
    value=analyze(sys.argv[1]);print(json.dumps({k:v for k,v in value.items() if k!='rows'},indent=2))
