"""Summarize measured evidence. Never convert kernel service into E2E shares."""
import collections
import csv
import gzip
import statistics
import numpy as np
from poc_looped_multigpu_kv_placement.contract import *
from poc_looped_multigpu_kv_placement.analysis import bootstrap,table


def read(path):return json.loads(path.read_text())


def category(name):
    lower=name.lower()
    if 'nccl' in lower or 'cross_device_reduce' in lower:return 'TP_COLLECTIVE_KERNEL'
    if 'reshape_and_cache' in lower:return 'KV_ROW_WRITE'
    if 'flashattn' in lower:return 'ATTENTION_INCLUDES_KV_READ'
    if 'nvjet' in lower or 'gemm' in lower:return 'PROJECTION_MLP_LMHEAD_GEMM'
    if 'rms_norm' in lower or 'silu' in lower:return 'NORM_ACTIVATION'
    if 'memcpy' in lower or 'memset' in lower:return 'MEMORY_OPERATION'
    return 'OTHER_KERNEL'


def union_ms(intervals):
    result=0.;end=None
    for a,b in sorted(intervals):
        if end is None or a>=end:result+=b-a;end=b
        elif b>end:result+=b-end;end=b
    return result/1000


def main():
    raw=ROOT/'results/raw';out=ROOT/'results'
    # Average technical repeats. Requests, not repeats, are independent units.
    huginn=[];huginn_groups={};seen=collections.defaultdict(list)
    for path in sorted((raw/'huginn_reference_native_v1').glob('P*.json')):
        row=read(path);cache=row['cache']
        seen[(row['prefix'],row['request_id'])].append(row)
        huginn.append({k:row[k] for k in ['prefix','request_id','repeat','prefill_ms','TPOT_ms','E2E_ms',
            'memory_allocated','memory_reserved','peak_allocated','peak_reserved']} | dict(
            logical_KV=cache['logical_unique_elements_bytes'],unique_KV_storage=cache['actual_unique_storage_bytes'],
            slots=cache['slot_count'],physical_GPU=5,steps=16,scope='Official HF single-GPU reference. Not native TP.'))
    for prefix in sorted({key[0] for key in seen}):
        values=[rows for (p,_),rows in seen.items() if p==prefix]
        huginn_groups[str(prefix)]={field:bootstrap([statistics.mean(r[field] for r in rows) for rows in values])
            for field in ['prefill_ms','TPOT_ms','E2E_ms','peak_allocated','peak_reserved']}
    table(out/'huginn_request_metrics.csv',huginn)
    write(out/'huginn_summary.json',dict(groups=huginn_groups,repeat_output_parity=all(
        len({tuple(r['output_ids']) for r in rows})==1 for rows in seen.values()),
        note='Same seed per request. Three technical repeats. Eight independent requests.'))
    # Native TP output differences remain descriptive, not quality scores.
    native=collections.defaultdict(list)
    for directory in sorted(raw.glob('ouro_tp*_native_v1')):
        for path in directory.glob('P*.repeat*.json'):
            if path.name.endswith('.memory.json'):continue
            row=read(path)
            for request in row['requests']:
                native[(row['tp'],row['prefix'],row['concurrency'],request['request_id'])].append((request['output_ids'],row['wall_ms']))
    pairs=[]
    for prefix in [1024,4096,8192,16384,32704]:
        for tp in [2,4]:
            changes=[];parity=[];first_differences=[]
            for key,rows in native.items():
                if key[:3]!=(1,prefix,1):continue
                other=native[(tp,prefix,1,key[3])]
                a=statistics.mean(r[1] for r in rows);b=statistics.mean(r[1] for r in other)
                changes.append((1-b/a)*100)
                x=rows[0][0];y=other[0][0];parity.append(x==y)
                if x!=y:first_differences.append(next(i for i,(v,w) in enumerate(zip(x,y)) if v!=w))
            pairs.append(dict(prefix=prefix,arm=f'TP{tp} vs TP1',paired_E2E_reduction_percent=bootstrap(changes),
                same_64_output_requests=sum(parity),requests=len(parity),first_difference_zero_based=first_differences))
    write(out/'native_correctness_and_pairs.json',dict(pairs=pairs,same_config_repeat_parity=all(
        len({tuple(r[0]) for r in rows})==1 for rows in native.values()),
        interpretation='Different TP accumulation can change greedy outputs. Fixed output budget is not benchmark accuracy.'))
    # Replica controls have one controller wall interval for the entire finite cohort.
    replica=[]
    for directory in raw.glob('ouro_replicas_tp*_dp*_v1'):
        for path in sorted(directory.glob('P*.repeat*.json')):
            row=read(path);engine_ms=[(r['metrics']['last_token_ts']-r['metrics']['queued_ts'])*1000
                for actor in row['actors'] for r in actor['requests']]
            replica.append(dict(source=directory.name,prefix=row['prefix'],concurrency=row['concurrency'],
                repeat=row['repeat'],tp=row['tp'],dp=row['dp'],wall_ms=row['controller_wall_ms'],
                aggregate_TPS=row['aggregate_TPS'],request_p95_engine_ms=float(np.quantile(engine_ms,.95)),
                request_count=len(engine_ms),scope='Finite-cohort wall includes gate/result IPC. Not B1 latency.'))
    table(out/'replica_throughput.csv',replica)
    throughput=collections.defaultdict(list)
    with (out/'capacity_frontier.csv').open() as stream:
        for row in csv.DictReader(stream):
            throughput[(f'TP{row["tp"]}',int(row['prefix']),int(row['concurrency']),int(row['repeat']))].append(
                (float(row['wall_ms']),int(row['requests'])))
    for row in replica:
        throughput[(f'TP{row["tp"]}xDP{row["dp"]}',row['prefix'],row['concurrency'],row['repeat'])].append(
            (row['wall_ms'],row['request_count']))
    cohort=[]
    for (layout,prefix,concurrency,repeat),batches in sorted(throughput.items()):
        wall=sum(v[0] for v in batches);count=sum(v[1] for v in batches)
        cohort.append(dict(layout=layout,prefix=prefix,offered_concurrency=concurrency,repeat=repeat,
            completed_requests=count,controller_wall_ms=wall,output_tokens=count*64,aggregate_TPS=count*64/(wall/1000),
            note='Sum actual controller walls across waves. Three technical repeats are not independent request samples.'))
    table(out/'cohort_throughput.csv',cohort)
    # Observer validity, separated from source attribution and kernel service.
    validity={};loop_rows=[];kernel_rows=[];kernel_summary=[];forward_services=[]
    for directory in [raw/'ouro_tp4_kernel_profile_v1',raw/'ouro_tp4_eager_source_v2']:
        path=directory/'profile_observation.json'
        if not path.exists():continue
        obs=read(path);baseline=statistics.mean(obs['clean_ms']);inflation=(obs['profile_ms']/baseline-1)*100
        validity[directory.name]=dict(clean_ms=obs['clean_ms'],profile_ms=obs['profile_ms'],
            observer_percent=inflation,output_parity=obs['output_parity'],
            quantitative_validity='UNVALIDATED' if inflation>10 else 'DIAGNOSTIC_ONLY',
            limitation='One request, four output tokens. Not the 64-token headline. Eager is a different execution lane.')
        if obs['eager']:
            for rank,events in enumerate(obs['trace']):
                groups=collections.defaultdict(list)
                for event in events:groups[(event['query_rows'],event['loop'],event['category'])].append(event['duration_ms'])
                for (rows,loop,cat),times in groups.items():loop_rows.append(dict(rank=rank,query_rows=rows,
                    loop=loop,category=cat,event_count=len(times),service_or_interval_ms=sum(times),
                    observer_percent=inflation,validity='UNVALIDATED' if inflation>10 else 'EAGER_DIAGNOSTIC',
                    note='Nested reduction is not additive. Not clean E2E or physical bandwidth.'))
    table(out/'loop_metrics.csv',loop_rows)
    for path in sorted((raw/'ouro_tp4_kernel_profile_v1').glob('torch_events.rank*.json.gz')):
        rank=int(path.name.split('rank')[1].split('.')[0])
        with gzip.open(path,'rt') as stream:events=json.load(stream)
        gpu=[e for e in events if e['device_type']=='DeviceType.CUDA'];groups=collections.defaultdict(list)
        names=collections.defaultdict(list)
        for event in gpu:groups[category(event['name'])].append(event);names[event['name']].append(event)
        for name,records in names.items():kernel_rows.append(dict(rank=rank,name=name,category=category(name),
            count=len(records),service_ms=sum(e['end_us']-e['start_us'] for e in records)/1000))
        kernel_summary.append(dict(rank=rank,total_kernel_service_ms=sum(e['end_us']-e['start_us'] for e in gpu)/1000,
            union_GPU_busy_ms=union_ms([(e['start_us'],e['end_us']) for e in gpu]),
            GPU_execution_span_ms=(max(e['end_us'] for e in gpu)-min(e['start_us'] for e in gpu))/1000,
            category_service_ms={cat:sum(e['end_us']-e['start_us'] for e in records)/1000 for cat,records in groups.items()},
            counts={cat:len(records) for cat,records in groups.items()},
            limitation='Per-rank diagnostic kernel service. No cross-rank clock alignment. Not E2E percentage.'))
        markers=sorted(e['start_us'] for e in gpu if e['name']=='_compute_slot_mappings_kernel')
        if len(markers)==4:
            for i,start in enumerate(markers):
                stop=markers[i+1] if i<3 else max(e['end_us'] for e in gpu)+1
                chosen=[e for e in gpu if start<=e['start_us']<stop]
                for cat in sorted({category(e['name']) for e in chosen}):
                    subset=[e for e in chosen if category(e['name'])==cat]
                    forward_services.append(dict(rank=rank,forward=i,query_rows=16384 if i==0 else 1,
                        phase='PREFILL' if i==0 else 'CACHED_DECODE',category=cat,count=len(subset),
                        kernel_service_ms=sum(e['end_us']-e['start_us'] for e in subset)/1000,
                        label='KERNEL_CLASSIFIED',boundary='_compute_slot_mappings_kernel once per native forward',
                        limitation='Service, not non-overlapped E2E share. KV read remains fused with attention.'))
    table(out/'kernel_services.csv',kernel_rows);write(out/'kernel_summary.json',kernel_summary)
    table(out/'forward_kernel_services.csv',forward_services)
    event_dir=raw/'ouro_tp4_outer_events_v1';event_pairs=collections.defaultdict(dict);outer=[]
    for path in sorted(event_dir.glob('*.json')):
        if not path.name.startswith(('coding_','math_','document_')):continue
        row=read(path);event_pairs[(row['request_id'],row['repeat'])][row['mode']]=row
        if row['mode']=='events':
            for rank,events in enumerate(row['trace']):
                for i,event in enumerate(events):outer.append(dict(request_id=row['request_id'],repeat=row['repeat'],
                    rank=rank,event_index=i,**event))
    if event_pairs:
        by_request=collections.defaultdict(list)
        for (request,_),arms in event_pairs.items():
            if len(arms)==2:by_request[request].append((arms['events']['wall_ms']/arms['clean']['wall_ms']-1)*100)
        if by_request:validity['outer_events']=dict(observer_percent=bootstrap([statistics.mean(v) for v in by_request.values()]),
            parity=all(row.get('parity',False) for arms in event_pairs.values() for row in arms.values()),
            note='Actual per-forward model/sampler GPU intervals. No multiplication of one replay by token count.')
    table(out/'outer_events.csv',outer);write(out/'observer_validity.json',validity)
    print('Wrote request-level statistics, output checks, observer checks, and separated service summaries.')


if __name__=='__main__':main()
