"""Separate worker emission gaps from coordinator delivery increments."""
import bisect
import collections
import json
import time
import numpy as np
from .common import ROOT,REPO,write,digest,preserve
from .registry import historical_runs,current_runs
from .reproduction import records,intervals
from .failure_analysis import save_csv


def token_times(events,key,n):
    result=[];previous=0
    for event in events:
        count=event['count'];assert previous<count
        result.extend([event[key]]*(count-previous));previous=count
    assert len(result)==n
    return np.asarray(result,float)


def analyze():
    preserve();starts=[];requests=[];violations=[];hashes={}
    selected=[row for row in historical_runs() if row['family'] in
        ('Historical_Main400_Natural','Historical_Heldout192_Natural','Historical_Heldout192_EqualReplay')]
    selected+=current_runs()
    for item in selected:
        folder=item['root']/'raw'/item['name'];raw=records(folder/'requests.jsonl')
        preparation=folder/'actual_queries_decode.json'
        query=json.loads(preparation.read_text())['rows'] if preparation.exists() else []
        prefill_times=sorted(row['time'] for row in query if row['API_prefilling'])
        meta={key:value for key,value in item.items() if key not in ('root','label_scope')}
        client_intervals=[];worker_intervals=[];local=[]
        for row in raw:
            n=row['output_tokens'];events=row['token_events']
            c=token_times(events,'time',n);w=token_times(events,'worker_time',n)
            assert np.all(np.diff(w)>=-1e-6)
            lag=c-w;assert np.all(lag>=-1e-6)
            ci=1000*np.diff(c);wi=1000*np.diff(w);increment=1000*np.diff(lag)
            assert np.allclose(ci,wi+increment,atol=1e-6)
            assert np.allclose(ci,intervals(events,n),atol=1e-6)
            cp=bool(np.all(ci<=34));wp=bool(np.all(wi<=34))
            category='BOTH_PASS' if cp and wp else 'CLIENT_FAIL_WORKER_PASS' if not cp and wp else 'CLIENT_PASS_WORKER_FAIL' if cp else 'BOTH_FAIL'
            value=dict(meta,request_id='q'+str(row['sample_id']),prompt_id=row['prompt_id'],sample_id=row['sample_id'],
                output_tokens=n,client_all_ITL_pass=cp,worker_emission_all_gap_pass=wp,classification=category,
                maximum_client_ITL_ms=float(ci.max()) if len(ci) else 0.,
                maximum_worker_gap_ms=float(wi.max()) if len(wi) else 0.,
                maximum_delivery_lag_ms=float(lag.max()*1000),TTFT_ms=row['TTFT_ms'])
            requests.append(value);local.append(value);client_intervals.extend(ci);worker_intervals.extend(wi)
            for index in np.flatnonzero((ci>34)|(wi>34)):
                left,right=w[index],w[index+1]
                count=bisect.bisect_right(prefill_times,right)-bisect.bisect_left(prefill_times,left)
                violations.append(dict(meta,request_id=value['request_id'],prompt_id=row['prompt_id'],interval_index=int(index),
                    output_tokens=n,client_ITL_ms=float(ci[index]),worker_gap_ms=float(wi[index]),
                    delivery_lag_increment_ms=float(increment[index]),delivery_lag_before_ms=float(lag[index]*1000),
                    delivery_lag_after_ms=float(lag[index+1]*1000),
                    D_prefill_prepare_rows_inside_worker_gap=count if preparation.exists() else 'NOT_INSTRUMENTED',
                    worker_before=left,worker_after=right,
                    scope='Client gap equals worker gap plus change in delivery lag. Preparation events are host observations,not exclusive kernel overlap.'))
        categories=[]
        for name in ('BOTH_PASS','CLIENT_FAIL_WORKER_PASS','CLIENT_PASS_WORKER_FAIL','BOTH_FAIL'):
            rows=[row for row in local if row['classification']==name]
            categories.append(dict(classification=name,requests=len(rows),output_tokens=sum(row['output_tokens'] for row in rows)))
        starts.append(dict(meta,request_count=len(raw),interval_count=len(client_intervals),categories=categories,
            client_ITL_p99_ms=float(np.percentile(client_intervals,99)),worker_gap_p99_ms=float(np.percentile(worker_intervals,99)),
            client_max_ITL_ms=max(client_intervals,default=0.),worker_max_gap_ms=max(worker_intervals,default=0.)))
        hashes[str((folder/'requests.jsonl').relative_to(REPO))]=digest(folder/'requests.jsonl')
        if preparation.exists():hashes[str(preparation.relative_to(REPO))]=digest(preparation)
    result=dict(time=time.time(),passed=True,starts=starts,source_files_SHA256=hashes,
        evidence='TRACE_ANALYSIS',scope='Worker emission and coordinator delivery are observed at separate host boundaries. '
            'Their algebraic gap decomposition does not identify CUDA kernel,DMA,engine step or Python GC causes. '
            'Worker token-span average is not reported as request TPOT because final completion uses another boundary. '
            'A client-only threshold crossing locates an observed delivery increment,not a deployable zero-delivery intervention. '
            'Co-delivered tokens retain zero intervals. Actual Natural output work and equal-output replay stay separate.')
    write(ROOT/'analysis/worker_client_decode_gap_analysis.json',result)
    for stem,rows in (('worker_client_decode_per_start',starts),('worker_client_decode_requests',requests),
            ('worker_client_decode_violations',violations)):
        save_csv(ROOT/'figures'/(stem+'.csv'),rows)
    preserve();return result


if __name__=='__main__':
    value=analyze();print('WORKER_CLIENT_DECODE_ANALYSIS',len(value['starts']),'starts',flush=True)
