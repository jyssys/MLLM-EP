"""Keep client delivery, worker service and finite-cohort throughput separate."""
import json
import numpy as np
from .common import ROOT,write


def percentile(values,p):
    return float(np.percentile(values,p)) if values else None


def delivery_intervals(events,output_tokens):
    assert events and events[-1]['count']==output_tokens
    assert events[0]['count']>=1
    # Co-delivered tokens have zero client-visible separation. Their GPU
    # generation intervals remain unknown and are not inferred here.
    intervals=[0.]*(events[0]['count']-1)
    for previous,current in zip(events,events[1:]):
        emitted=current['count']-previous['count'];assert emitted>0
        assert current['time']>=previous['time']
        intervals.extend([1000*(current['time']-previous['time'])]+[0.]*(emitted-1))
    assert len(intervals)==output_tokens-1
    return intervals


def summarize(name):
    folder=ROOT/'raw'/name;cfg=json.loads((folder/'configuration.json').read_text())
    rows=[json.loads(line) for line in (folder/'requests.jsonl').read_text().splitlines()]
    window=json.loads((folder/'window.json').read_text());duration=window['end']-window['start']
    expected=cfg['count'];assert len(rows)==expected
    all_itl=[];good=[];per_request=[]
    for row in rows:
        events=row['token_events'];intervals=delivery_intervals(events,row['output_tokens'])
        all_itl.extend(intervals)
        ttft_pass=row['TTFT_ms']<=5000;itl_pass=all(x<=34 for x in intervals)
        tpot_pass=row['TPOT_ms'] is None or row['TPOT_ms']<=33
        slo=ttft_pass and itl_pass and tpot_pass
        if slo:good.append(row)
        per_request.append(dict(prompt_id=row['prompt_id'],task=row.get('task'),SLO_pass=slo,
            TTFT_pass=ttft_pass,every_ITL_pass=itl_pass,TPOT_pass=tpot_pass,decode_pass=itl_pass and tpot_pass,
            maximum_ITL_ms=max(intervals,default=0),output_tokens=row['output_tokens']))
    output_tokens=sum(r['output_tokens'] for r in rows);tpots=[r['TPOT_ms'] for r in rows if r['TPOT_ms'] is not None]
    dispatch_file=folder/'offered_requests.jsonl'
    dispatch=[json.loads(line) for line in dispatch_file.read_text().splitlines()] if dispatch_file.exists() else []
    enqueue_times=[r['enqueue'] for r in dispatch]
    dispatch_lags=[1000*(r['enqueue']-r['arrival']) for r in dispatch]
    result=dict(name=name,method=cfg['method'],n=len(rows),independent_server_starts=1,
        TTFT_p95_ms=percentile([r['TTFT_ms'] for r in rows],95),TTFT_mean_ms=float(np.mean([r['TTFT_ms'] for r in rows])),
        TTFT_p50_ms=percentile([r['TTFT_ms'] for r in rows],50),TTFT_p99_ms=percentile([r['TTFT_ms'] for r in rows],99),
        ITL_p99_ms=percentile(all_itl,99),ITL_p95_ms=percentile(all_itl,95),mean_TBT_ms=float(np.mean(all_itl)) if all_itl else None,
        max_ITL_ms=max(all_itl,default=0),ITL_interval_count=len(all_itl),
        first_event_multi_token_requests=sum(r['token_events'][0]['count']>1 for r in rows),
        token_delivery_scope='Client-visible co-delivered tokens have zero interval. No per-token GPU generation timestamp is inferred.',
        TPOT_p95_ms=percentile(tpots,95),TPOT_p50_ms=percentile(tpots,50),TPOT_p99_ms=percentile(tpots,99),
        E2E_p95_ms=percentile([r['E2E_ms'] for r in rows],95),output_tokens=output_tokens,
        generated_output_tokens_per_second=output_tokens/duration,completed_requests_per_second=len(rows)/duration,
        Joint_SLO_attainment=len(good)/len(rows),SLO_qualified_output_tokens_per_second=sum(r['output_tokens'] for r in good)/duration,
        TTFT_SLO_attainment=sum(r['TTFT_pass'] for r in per_request)/len(rows),
        every_ITL_SLO_attainment=sum(r['every_ITL_pass'] for r in per_request)/len(rows),
        TPOT_SLO_attainment=sum(r['TPOT_pass'] for r in per_request)/len(rows),
        decode_SLO_attainment=sum(r['decode_pass'] for r in per_request)/len(rows),
        offered_QPS=cfg.get('offered_QPS'),actual_arrival_QPS=(expected-1)/(max(r['arrival'] for r in rows)-min(r['arrival'] for r in rows)) if expected>1 and not cfg.get('serial') else None,
        actual_dispatch_QPS=(len(enqueue_times)-1)/(max(enqueue_times)-min(enqueue_times)) if len(enqueue_times)>1 else None,
        dispatch_lag_p95_ms=percentile(dispatch_lags,95),dispatch_lag_max_ms=max(dispatch_lags,default=None),
        arrival_rate_definition='actual_arrival_QPS is the realized scheduled Poisson-trace rate. actual_dispatch_QPS uses coordinator enqueue timestamps. TTFT includes dispatch lag.',
        measurement_seconds=duration,last_arrival_seconds=window['last_arrival']-window['start'],
        finite_cohort=True,sustainable_capacity=False,
        output_contract='DIAGNOSTIC_FIXED_LENGTH_REPLAY' if cfg.get('diagnostic_controlled_output_replay') else 'NATURAL_EOS',
        SLO=dict(TTFT_ms=5000,every_ITL_ms=34,request_TPOT_ms=33),
        timing_scope='Scheduled open-loop arrival to coordinator delivery. No HTTP. First Native interval includes KV handoff if P first token is reused.',
        per_request=per_request,by_task={})
    for task in sorted({r.get('task','') for r in rows}):
        part=[r for r in rows if r.get('task','')==task]
        good_ids={r['prompt_id'] for r in good}
        task_itl=[]
        for row in part:
            task_itl.extend(delivery_intervals(row['token_events'],row['output_tokens']))
        result['by_task'][task]=dict(n=len(part),TTFT_p95_ms=percentile([r['TTFT_ms'] for r in part],95),
            TTFT_mean_ms=float(np.mean([r['TTFT_ms'] for r in part])),
            mean_TBT_ms=float(np.mean(task_itl)) if task_itl else None,ITL_p99_ms=percentile(task_itl,99),
            TPOT_p95_ms=percentile([r['TPOT_ms'] for r in part if r['TPOT_ms'] is not None],95),
            output_tokens=sum(r['output_tokens'] for r in part),Joint_SLO_attainment=sum(r['prompt_id'] in good_ids for r in part)/len(part),
            SLO_components={key:sum(r[key] for r in per_request if r['task']==task)/len(part) for key in ('TTFT_pass','every_ITL_pass','TPOT_pass','decode_pass')},
            output_tokens_per_global_window_second=sum(r['output_tokens'] for r in part)/duration,
            scope='Task-specific latency in the same Mixed workload. Throughput contribution uses the global window, not an isolated task capacity.')
    write(ROOT/'analysis'/(name+'_metrics.json'),result);return result


if __name__=='__main__':
    import sys
    print(json.dumps({k:v for k,v in summarize(sys.argv[1]).items() if k not in ('per_request','by_task')},indent=2))
