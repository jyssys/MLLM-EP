"""Decompose observed request boundaries without inferring physical GPU idle."""
import collections
import json
import numpy as np
from .common import ROOT,write
from .qualification import records


def stats(values):
    values=list(values)
    return dict(n=len(values),mean=float(np.mean(values)),p50=float(np.percentile(values,50)),
        p90=float(np.percentile(values,90)),p95=float(np.percentile(values,95)),
        p99=float(np.percentile(values,99)),max=float(max(values))) if values else dict(n=0,status='NOT_INSTRUMENTED')


def time_weighted(values,key,window):
    points=[]
    for index,row in enumerate(values):
        if key not in row:continue
        left=max(row['time'],window['start'])
        right=min(values[index+1]['time'] if index+1<len(values) else window['end'],window['end'])
        if right>left:points.append((row[key],right-left))
    if not points:return dict(status='NOT_INSTRUMENTED')
    x=np.asarray([v for v,_ in points],dtype=float);w=np.asarray([v for _,v in points],dtype=float)
    order=np.argsort(x);cdf=np.cumsum(w[order])/w.sum()
    return dict(mean=float(np.sum(x*w)/w.sum()),p95=float(x[order][np.searchsorted(cdf,.95)]),
        max=float(max(x)),covered_seconds=float(w.sum()),
        scope='Piecewise-constant integration of observed scheduler states. Not exact state-transition timing or GPU activity.')


def analyze(name):
    folder=ROOT/'raw'/name;cfg=json.loads((folder/'configuration.json').read_text());rows=records(folder/'requests.jsonl')
    data=json.loads((ROOT/'data'/cfg['dataset']).read_text())[:cfg['count']]
    queries={};queues={};geometry={};jit={};queue_boundaries={};refusals={}
    for role in ('prefill','decode','replica0','replica1'):
        file=folder/f'actual_queries_{role}.json'
        if not file.exists():continue
        part=json.loads(file.read_text())['rows'];queries[role]=part
        groups=collections.defaultdict(list)
        for row in part:
            # The old witness sampled time separately per row. Its batches cannot be deduplicated by time.
            if 'batch_id' in row:groups[row['batch_id']].append(row)
        prefill_counts=[sum(q['start']<q['original_input_tokens'] for q in g) for g in groups.values()]
        decode_counts=[sum(q['start']>=q['original_input_tokens'] for q in g) for g in groups.values()]
        geometry[role]=dict(actual_request_batch_histogram=dict(collections.Counter(g[0]['requests'] for g in groups.values())),
            actual_prefill_request_batch_histogram=dict(collections.Counter(n for n in prefill_counts if n)),
            actual_decode_request_batch_histogram=dict(collections.Counter(n for n in decode_counts if n)),
            mixed_prefill_decode_iterations=sum(p>0 and d>0 for p,d in zip(prefill_counts,decode_counts)),
            prefill_only_iterations=sum(p>0 and d==0 for p,d in zip(prefill_counts,decode_counts)),
            decode_only_iterations=sum(d>0 and p==0 for p,d in zip(prefill_counts,decode_counts)),
            padded_query_rows=sum(g[0]['padded_queries'] for g in groups.values()),
            valid_query_rows=sum(q['count'] for q in part),
            measured_batches=len(groups),grouped_batch_status='MEASURED' if groups else 'NOT_INSTRUMENTED_BATCH_ID',
            request_weighted_batch_histogram=dict(collections.Counter(r['requests'] for r in part)),
            actual_query_lengths=stats(r['count'] for r in part if r['start']<r['original_input_tokens']),
            scope='Request-weighted histograms are not iteration-weighted histograms. Actual query lengths differ from scheduler token limits.')
        schedule=folder/f'scheduler_queues_{role}.json'
        queues[role]=json.loads(schedule.read_text())['rows'] if schedule.exists() else json.loads((folder/f'engine_steps_{role}.json').read_text())['queues']
        queue_boundaries[role]='SCHEDULER_EXIT_BEFORE_WORK' if schedule.exists() else 'POST_ENGINE_STEP_INCOMPLETE_SKIPPED_QUEUE_SCHEMA'
        file=folder/f'JIT_events_{role}.json'
        jit[role]=json.loads(file.read_text())['rows'] if file.exists() else 'NOT_INSTRUMENTED'
        file=folder/f'KV_allocation_refusals_{role}.json'
        refusals[role]=json.loads(file.read_text())['rows'] if file.exists() else 'NOT_INSTRUMENTED'
    paths=[]
    if cfg['method']=='Native':
        handoffs={r['request_id']:r for r in records(folder/'handoffs.jsonl')}
        admissions=records(folder/'admissions.jsonl') if (folder/'admissions.jsonl').exists() else []
        imported={}
        file=folder/'NIXL_completion_decode.json'
        if file.exists():
            for event in json.loads(file.read_text())['rows']:
                for rid in event['finished_recving']:imported[rid.split('-')[0]]=event['time']
        for row in rows:
            rid=f'q{row["sample_id"]}';source=[q for q in queries['prefill'] if q['request_id']==rid]
            dest=[q for q in queries['decode'] if q['request_id']==rid]
            assert source and dest;first_P=min(q['time'] for q in source);first_D=min(q['time'] for q in dest)
            ready=handoffs[rid]['source_ready'];cursor=0;coverage=True
            for q in source:
                coverage &= q['start']==cursor;cursor+=q['count']
            coverage &= cursor==row['input_tokens']
            accepts=[r for r in admissions if r['request_id']==rid and r['role']=='decode']
            p_accepts=[r for r in admissions if r['request_id']==rid and r['role']=='prefill']
            path=dict(request_id=rid,prompt_id=row['prompt_id'],task=row.get('task'),
                P_queue_wait_ms=1000*(first_P-row['arrival']),P_first_prepare_to_ready_ms=1000*(ready-first_P),
                Arrival_to_P_accepted_ms=1000*(p_accepts[0]['worker_accepted']-row['arrival']) if p_accepts else None,
                P_accepted_to_P_prepare_ms=1000*(first_P-p_accepts[0]['worker_accepted']) if p_accepts else None,
                Ready_to_coordinator_ms=1000*(handoffs[rid]['received']-ready),
                Ready_to_D_prepare_ms=1000*(first_D-ready),D_prepare_to_first_delivery_ms=1000*(row['first_token']-first_D),
                TTFT_ms=row['TTFT_ms'],P_query_coverage_passed=bool(coverage),
                D_first_query_start=min(q['start'] for q in dest),
                D_original_prompt_query_rows=sum(max(0,min(q['count'],row['input_tokens']-q['start'])) for q in dest),
                D_total_query_rows=sum(q['count'] for q in dest),
                D_generation_query_rows=sum(q['count'] for q in dest if q['start']>=row['input_tokens']),
                D_post_prompt_rows_minus_output_tokens_after_first=sum(q['count'] for q in dest if q['start']>=row['input_tokens'])-(row['output_tokens']-1),
                Ready_to_D_accepted_ms=1000*(accepts[0]['worker_accepted']-ready) if accepts else None,
                Ready_to_NIXL_receive_notification_ms=1000*(imported[rid]-ready) if rid in imported else None,
                receive_notification_to_D_prepare_ms=1000*(first_D-imported[rid]) if rid in imported else None)
            paths.append(path)
    boundaries={key:stats(r[key] for r in paths if r[key] is not None) for key in
        ('P_queue_wait_ms','Arrival_to_P_accepted_ms','P_accepted_to_P_prepare_ms','P_first_prepare_to_ready_ms','Ready_to_coordinator_ms','Ready_to_D_prepare_ms',
         'D_prepare_to_first_delivery_ms','Ready_to_D_accepted_ms','Ready_to_NIXL_receive_notification_ms','receive_notification_to_D_prepare_ms')}
    queue_stats={role:{key:stats(r[key] for r in values if key in r) for key in
        ('waiting','skipped_waiting','waiting_total','remote_KV_waiting','running','active_decode','decode_ready_waiting','decode_ready_including_first_query','free_KV_blocks')} for role,values in queues.items()}
    window=json.loads((folder/'window.json').read_text())
    weighted={role:{key:time_weighted(values,key,window) for key in
        ('waiting_primary','waiting_total','remote_KV_waiting','running','active_decode','decode_ready_waiting','decode_ready_including_first_query')} for role,values in queues.items()}
    value=dict(name=name,method=cfg['method'],n=len(rows),input_tokens=stats(r['input_tokens'] for r in rows),
        output_tokens=stats(r['output_tokens'] for r in rows),boundaries=boundaries,paths=paths,
        queues=queue_stats,time_weighted_queues=weighted,queue_sampling_boundary=queue_boundaries,geometry=geometry,JIT_events=jit,
        KV_allocation_refusals=refusals,evidence='TRACE_ANALYSIS',
        P_query_coverage_passed=all(r['P_query_coverage_passed'] for r in paths) if paths else None,
        Native_D_only_last_prompt_query=all(r['D_original_prompt_query_rows']==1 for r in paths) if paths else None,
        scope='Host-observed boundaries and sampled scheduler states. Not exclusive GPU compute, physical idle, decode-safe headroom or sustainable capacity. Ready-to-D-prepare includes transport and admission. Raw queue summaries are event-weighted. time_weighted_queues integrates the sampled states explicitly.')
    write(ROOT/'analysis'/(name+'_timeline_analysis.json'),value);return value


if __name__=='__main__':
    import sys
    result=analyze(sys.argv[1]);print(json.dumps({k:v for k,v in result.items() if k not in ('paths','JIT_events')},indent=2))
