"""Reconstruct additive host paths and preserve non-additive transfer probes."""
import bisect
import collections
import json
import re
import time
import numpy as np
from .common import ROOT,REPO,initialize,preserve,write,digest
from .reproduction import records,reconstruct
from .registry import analysis_runs
from .failure_analysis import save_csv

STAGES=('Arrival_to_P_submit_ms','P_submit_to_accept_ms','P_accept_to_prepare_ms',
    'P_prepare_to_ready_ms','Ready_to_D_submit_ms','D_submit_to_accept_ms',
    'D_accept_to_prepare_ms','D_prepare_to_first_worker_ms','Worker_to_client_ms')
STAMPS=('arrival','P_submit','P_accept','P_prepare','P_ready','D_submit','D_accept','D_prepare','first_worker','first_client')


def external(value):
    if value is None:return None
    match=re.match(r'^(q\d+)(?:-|$)',value)
    return match[1] if match else None


def read_json_rows(path):return json.loads(path.read_text()).get('rows',[]) if path.exists() else []


def credit_observation(folder,cfg):
    path=folder/'credit_events.jsonl'
    if path.exists():return 'INSTRUMENTED',records(path)
    return ('UNBOUNDED_COORDINATOR_WINDOW' if cfg.get('continuous_source_credit_limit',0)==0 else 'NOT_INSTRUMENTED'),[]


def before(rows,point):
    times=[row['time'] for row in rows];index=bisect.bisect_right(times,point)-1
    return rows[index] if index>=0 else None


def trace(folder):
    cfg,metric,client=reconstruct(folder)
    raw={f'q{r["sample_id"]}':r for r in records(folder/'requests.jsonl')}
    submissions={(r['request_id'],r['role']):r['time'] for r in records(folder/'request_submissions.jsonl')}
    accepted={(r['request_id'],r['role']):r['worker_accepted'] for r in records(folder/'admissions.jsonl')}
    ready={r['request_id']:r for r in records(folder/'handoffs.jsonl')}
    query={};work=collections.defaultdict(list)
    for role in ('prefill','decode'):
        for row in read_json_rows(folder/f'actual_queries_{role}.json'):
            query.setdefault((row['request_id'],role),row['time'])
            work[row['request_id'],role].append(row)
    queues={role:read_json_rows(folder/f'scheduler_queues_{role}.json') for role in ('prefill','decode')}
    reads=collections.defaultdict(list);acks={};native_telemetry=collections.defaultdict(list)
    for row in read_json_rows(folder/'NIXL_host_setup_decode.json'):
        rid=external(row.get('request_id',''))
        if rid is None:continue
        if row.get('kind')=='read_blocks_host':reads[rid].append(row)
        if row.get('kind')=='NIXL_native_telemetry':native_telemetry[rid].append(row)
    for row in read_json_rows(folder/'NIXL_completion_decode.json'):
        for request in row.get('finished_recving',[]):
            rid=external(request)
            if rid is not None:acks.setdefault(rid,row['time'])
    credit=collections.defaultdict(list);credit_status,credit_rows=credit_observation(folder,cfg)
    for row in credit_rows:credit[row['request_id']].append(row)
    offered={row['request_id']:row for row in records(folder/'offered_requests.jsonl')} if (folder/'offered_requests.jsonl').exists() else {}
    rows=[]
    for rid,item in sorted(raw.items(),key=lambda r:int(r[0][1:])):
        assert item['token_events'][0].get('source')!='P_FIRST_SAMPLE','Separate first-token-reuse paths before additive D-first accounting'
        times=(item['arrival'],submissions[rid,'prefill'],accepted[rid,'prefill'],query[rid,'prefill'],
            ready[rid]['source_ready'],submissions[rid,'decode'],accepted[rid,'decode'],query[rid,'decode'],
            item['token_events'][0]['worker_time'],item['first_token'])
        parts={key:1000*(right-left) for key,left,right in zip(STAGES,times,times[1:])}
        assert all(value>=-1e-3 for value in parts.values()),(folder.name,rid,parts)
        assert abs(sum(parts.values())-item['TTFT_ms'])<1e-3
        row=dict(request_id=rid,sample_id=item['sample_id'],prompt_id=item['prompt_id'],task=item.get('task'),
            input_tokens=item['input_tokens'],output_tokens=item['output_tokens'],TTFT_ms=item['TTFT_ms'],
            **parts,boundaries=dict(zip(STAMPS,times)),Arrival_to_P_ready_ms=1000*(times[4]-times[0]),
            post_P_to_first_client_ms=1000*(times[-1]-times[4]),
            D_decode_client_residency_ms=1000*(item['finish']-item['first_token']),
            P_query_chunks=len([q for q in work[rid,'prefill'] if q['API_prefilling']]),
            D_query_chunks=len([q for q in work[rid,'decode'] if q['API_prefilling']]),
            actual_P_GPU_start='NOT_INSTRUMENTED',actual_P_GPU_end='NOT_INSTRUMENTED',
            actual_D_GPU_start='NOT_INSTRUMENTED',actual_D_GPU_end='NOT_INSTRUMENTED')
        for role in ('prefill','decode'):
            snapshot=before(queues[role],item['arrival'])
            row[role+'_queue_at_arrival']=dict(snapshot) if snapshot else None
            first=work[rid,role][0]
            row[role+'_first_query_geometry']={key:first.get(key) for key in
                ('requests','actual_batch_queries','padded_queries','count','start','API_prefilling')}
        row['credit_events']=credit.get(rid,[])
        row['source_credit_observation']=credit_status
        detected=offered.get(rid,{}).get('enqueue')
        acquisition=next((event['time'] for event in credit.get(rid,[]) if event['event']=='ACQUIRE_P_DISPATCH'),None)
        row['Coordinator_loop_detection_lateness_ms']=1000*(detected-times[0]) if detected is not None else None
        row['Loop_detection_to_P_submit_ms']=1000*(times[1]-detected) if detected is not None else None
        row['Loop_detection_to_credit_acquire_ms']=1000*(acquisition-detected) if acquisition is not None and detected is not None else None
        row['Credit_acquire_to_P_submit_ms']=1000*(times[1]-acquisition) if acquisition is not None else None
        row['coordinator_detection_scope']='The recorded enqueue field is the coordinator loop detection timestamp. It is not an HTTP ingress timestamp or exclusive credit wait.'
        if detected is not None:
            assert min(row['Coordinator_loop_detection_lateness_ms'],row['Loop_detection_to_P_submit_ms'])>=-1e-3
            assert abs(row['Coordinator_loop_detection_lateness_ms']+row['Loop_detection_to_P_submit_ms']-parts['Arrival_to_P_submit_ms'])<1e-3
        if reads.get(rid) and rid in acks:
            left=min(r['start'] for r in reads[rid]);right=max(r['end'] for r in reads[rid]);ack=acks[rid]
            component=[times[6],left,right,ack,times[7]]
            ordered=all(b>=a-1e-6 for a,b in zip(component,component[1:]))
            row['remote_read_breakdown']=dict(ordered=ordered,read_calls=len(reads[rid]),
                D_accept_to_read_start_ms=1000*(left-times[6]),read_host_enclosure_ms=1000*(right-left),
                read_end_to_receive_ACK_ms=1000*(ack-right),receive_ACK_to_D_prepare_ms=1000*(times[7]-ack),
                read_start=left,read_end=right,receive_ACK=ack,
                scope='Host READ enclosure and engine-poll ACK boundaries. Not DMA-exclusive time. Hidden import can have separate dependencies.')
        else:row['remote_read_breakdown']=dict(status='NOT_INSTRUMENTED_FOR_REQUEST')
        row['NIXL_native_telemetry']=native_telemetry.get(rid,[])
        rows.append(row)
    return cfg,metric,rows


def analyze():
    initialize();preserve();runs=[];all_rows=[];source_files={};unavailable=[]
    for item in analysis_runs():
        if item['label'] in ('Agg','Deflection'):
            unavailable.append(dict(family=item['family'],label=item['label'],name=item['name'],
                reason='Different colocated or whole-prefill-deflection DAG. No artificial P→D stages assigned.'));continue
        folder=item['root']/'raw'/item['name'];cfg,metric,rows=trace(folder)
        meta={key:value for key,value in item.items() if key!='root'}
        meta.update(seed=cfg['seed'],GPU_pair=cfg['gpus'],dataset_SHA256=cfg['dataset_SHA256'],
            source_SHA256=json.loads((folder/'dispatcher_result.json').read_text())['source_SHA256'])
        fields=(*STAGES,'Arrival_to_P_ready_ms','post_P_to_first_client_ms','D_decode_client_residency_ms','P_query_chunks','D_query_chunks')
        means={key:float(np.mean([row[key] for row in rows])) for key in fields}
        components=[row['remote_read_breakdown'] for row in rows if row['remote_read_breakdown'].get('ordered')]
        transfers={key:float(np.mean([row[key] for row in components])) for key in (
            'D_accept_to_read_start_ms','read_host_enclosure_ms','read_end_to_receive_ACK_ms','receive_ACK_to_D_prepare_ms')} if components else {}
        runs.append(dict(meta,n=len(rows),means=means,remote_read_ordered_requests=len(components),remote_read_means=transfers))
        extra=('Coordinator_loop_detection_lateness_ms','Loop_detection_to_P_submit_ms',
            'Loop_detection_to_credit_acquire_ms','Credit_acquire_to_P_submit_ms')
        runs[-1]['coordinator_credit_means']={key:float(np.mean([row[key] for row in rows if row[key] is not None]))
            if any(row[key] is not None for row in rows) else None for key in extra}
        all_rows.extend(dict(meta,**row) for row in rows)
        for filename in ('request_submissions.jsonl','admissions.jsonl','handoffs.jsonl','actual_queries_prefill.json',
                'actual_queries_decode.json','NIXL_host_setup_decode.json','NIXL_completion_decode.json',
                'offered_requests.jsonl','credit_events.jsonl'):
            path=folder/filename
            if path.exists():source_files[str(path.relative_to(REPO))]=digest(path)
    value=dict(time=time.time(),passed=True,starts=runs,unassigned_DAGs=unavailable,source_files_SHA256=source_files,
        stage_names=STAGES,evidence='TRACE_ANALYSIS',
        scope='Common host boundaries sum to each client TTFT. Prepare is not GPU start. P/D host enclosures include other requests and scheduler gaps. '
            'Transfer API/ACK sub-boundaries are not exclusive DMA. Do not add nested enclosures to the outer critical path. '
            'Absent GPU boundaries remain NOT_INSTRUMENTED. Native first-token source is explicitly checked. '
            'Historical workloads and source/cache policies stay separate. Additive means do not decompose p95.')
    write(ROOT/'analysis/request_critical_paths.json',value)
    save_csv(ROOT/'figures/request_critical_paths.csv',all_rows)
    compact_fields=('family','label','start','name','sample_id','prompt_id','input_tokens','output_tokens','TTFT_ms',
        *STAGES,'boundaries','Arrival_to_P_ready_ms','post_P_to_first_client_ms','remote_read_breakdown',
        'prefill_queue_at_arrival','decode_queue_at_arrival','prefill_first_query_geometry','decode_first_query_geometry',
        'source_credit_observation','Coordinator_loop_detection_lateness_ms','Loop_detection_to_P_submit_ms',
        'Loop_detection_to_credit_acquire_ms','Credit_acquire_to_P_submit_ms',
        'seed','GPU_pair','dataset_SHA256','source_SHA256')
    save_csv(ROOT/'figures/main400_request_critical_paths.csv',[
        {key:row[key] for key in compact_fields} for row in all_rows if row['family'] in
        ('Historical_Main400_Natural','Current_Main400_Natural')])
    save_csv(ROOT/'figures/host_stage_per_start.csv',runs)
    save_csv(ROOT/'figures/structurally_different_stage_DAGs.csv',unavailable)
    print('REQUEST_CRITICAL_PATH_PASS',len(runs),'starts',len(all_rows),'requests',flush=True)
    preserve();return value


if __name__=='__main__':analyze()
