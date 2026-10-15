"""Rebuild compact evidence from completed raw runs. Safe during collection."""
from collections import defaultdict,Counter
import gzip
import hashlib
import statistics
import numpy as np
from poc_looped_recurrence_tp_4models.contract import *
from poc_looped_recurrence_tp_4models.analysis import read,table,bootstrap,category,union_ms


def main():
    out=ROOT/'results';raw=out/'raw';measures=[];requests=[];owners=[];validity={};parity=[]
    kernels=[];services=[];tapes={};outer_rows=[];runtime=[];graph_counts=[];coverage=[];layer_attention=[]
    for directory in sorted(raw.glob('*_v1')):
        if 'smoke' in directory.name:continue
        status=directory/'status.json'
        if not status.exists() or not read(status).get('success'):continue
        state=read(status);model=state['model'];family=state['family'];R=state['r'];TP=state['tp']
        key=dict(model=model,family=family,R=R,TP=TP)
        depth=48*R if model=='ouro' else {'qwen3':28,'qwen25':36}[model]
        ownership=read(directory/'ownership.json');by_request=defaultdict(list)
        for rank,r in enumerate(ownership):
            geom=r['geometry'][0];heads=geom['KV_heads'];d=geom['head_dim']
            owners.append(key|dict(rank=rank,GPU_uuid=r['GPU_uuid'],physical_GPU=GPUS[rank],
                physical_layers=r['physical_layers'],logical_depth=depth,cache_slots=r['cache_slots'],
                Q_heads_local=geom['Q_heads'],KV_heads_local=heads,head_dim=d,
                unique_weight_bytes=r['unique_parameter_storage_bytes'],KV_pool_bytes=r['unique_KV_pool_storage_bytes'],
                KV_pool_blocks=r['KV_pool_blocks'],model_runner=r['model_runner'],
                K_weight_sha256=geom['K_weight_sha256'],V_weight_sha256=geom['V_weight_sha256']))
        for path in sorted(directory.glob('P*.repeat*.json')):
            if path.name.endswith('.memory.json'):continue
            row=read(path);metrics=row['metrics'];memory=read(path.with_name(path.stem+'.memory.json'))
            value=key|dict(prefix=row['prefix'],request_id=row['request_id'],domain=row['domain'],repeat=row['repeat'],
                output_tokens=len(row['output_ids']),E2E_ms=row['wall_ms'],
                prefill_ms=(metrics['first_token_ts']-metrics['scheduled_ts'])*1000,
                TTFT_client_metric_ms=metrics['first_token_latency']*1000,
                TPOT_ms=(metrics['last_token_ts']-metrics['first_token_ts'])*1000/63,
                output_TPS=64000/row['wall_ms'],
                peak_allocated_bytes=max(r['peak_allocated'] for r in memory),
                peak_reserved_bytes=max(r['peak_reserved'] for r in memory),
                cached_tokens=row['cached_tokens'],logical_depth=depth,
                logical_KV_bytes_per_rank=depth*(row['prefix']+63)*2*ownership[0]['geometry'][0]['KV_heads']*128*2,
                scope='PRIMARY_N8' if row['prefix']!=32704 else '32K_CONFIRMATION_N3')
            measures.append(value);by_request[(row['prefix'],row['request_id'])].append(value)
        for (prefix,request),repeats in by_request.items():
            fields=['E2E_ms','prefill_ms','TPOT_ms','output_TPS','peak_allocated_bytes','peak_reserved_bytes','logical_KV_bytes_per_rank']
            requests.append(key|dict(prefix=prefix,request_id=request,domain=repeats[0]['domain'],technical_repeats=len(repeats),
                logical_depth=depth,**{f:statistics.mean(r[f] for r in repeats) for f in fields}))
        for path in sorted(directory.glob('parity_check_*.json')):
            for check in read(path):parity.append(key|dict(prefix=int(path.stem.split('P')[1]),**check))
        coverage.append(key|dict(directory=directory.name,completed_request_repeats=len(measures)-sum(c['completed_request_repeats'] for c in coverage)))
        obs=read(directory/'kernel_profile.json')
        clean64=[r['E2E_ms'] for r in by_request[(16384,'coding_00')]]
        outer=read(directory/'outer_and_counts.json')
        overhead=(outer['result']['wall_ms']/statistics.mean(clean64)-1)*100
        validity[directory.name]=dict(kernel_observer_percent=obs['observer_percent'],
            kernel_timing='UNVALIDATED' if obs['observer_percent']>10 else 'DIAGNOSTIC_SINGLE_PROFILE',
            outer_observer_percent=overhead,outer_timing='UNVALIDATED' if overhead>10 else 'DIAGNOSTIC_SINGLE_OBSERVER',
            clean_64_ms=clean64,profile_4_ms=obs['profile_ms'],clean_4_ms=obs['clean_ms'],
            limitation='Signed one-profile differences include drift. No request-level observer CI. Clean wall is separate.')
        for rank,events in enumerate(outer['outer']):
            for index,event in enumerate(events):outer_rows.append(key|dict(rank=rank,event_index=index,**event))
            runtime.append(key|dict(rank=rank,execute_model_API_calls=outer['counts'][rank]['model_forwards'],
                actual_nonempty_forwards=sum(e['category']=='execute_model' and (e['scheduled_rows'] or 0)>0 for e in events),
                empty_execute_calls=sum(e['category']=='execute_model' and e['scheduled_rows']==0 for e in events),
                graph_replays=outer['counts'][rank]['graph_replays'],sampling_calls=outer['counts'][rank]['sampling_calls'],
                expected_logical_attention_executions=64*depth,
                raw_graph_node_count='NOT_AVAILABLE_IN_PRIMARY_KEEP_GRAPH_FALSE'))
        for path in sorted(directory.glob('torch_events.rank*.json.gz')):
            rank=int(path.name.split('rank')[1].split('.')[0])
            with gzip.open(path,'rt') as stream:events=json.load(stream)
            gpu=sorted([e for e in events if 'CUDA' in e['device_type']],key=lambda e:(e['start_us'],e['end_us']))
            markers=[e['start_us'] for e in gpu if e['name']=='_compute_slot_mappings_kernel']
            assert len(markers)==4,(directory,rank,len(markers))
            for forward,start in enumerate(markers):
                stop=markers[forward+1] if forward<3 else float('inf')
                selected=[e for e in gpu if start<=e['start_us']<stop]
                writes=[e['start_us'] for e in selected if 'reshape_and_cache' in e['name']]
                assert len(writes)==depth,(directory,rank,forward,len(writes),depth)
                for logical,begin in enumerate(writes):
                    end=writes[logical+1] if logical+1<len(writes) else stop
                    attention=[e for e in selected if begin<=e['start_us']<end and category(e['name'])=='ATTENTION_WITH_KV_READ']
                    layer_attention.append(key|dict(rank=rank,forward=forward,phase='PREFILL' if forward==0 else 'CACHED_DECODE',
                        logical_layer=logical,loop=logical//48 if model=='ouro' else 0,
                        physical_layer=logical%48 if model=='ouro' else logical,
                        attention_service_ms=sum(e['end_us']-e['start_us'] for e in attention)/1000,
                        attention_kernels=len(attention),KV_write_start_us=begin,
                        label='KERNEL_CLASSIFIED. Includes fused KV read. Not full-layer latency or critical-path share.'))
                groups=defaultdict(list)
                for event in selected:groups[event['name']].append(event)
                tape=[e['name'] for e in selected if 'memcpy' not in e['name'].lower() and 'memset' not in e['name'].lower()]
                collective=[name for name in tape if category(name)=='TP_COLLECTIVE']
                graph_counts.append(key|dict(rank=rank,forward=forward,phase='PREFILL' if forward==0 else 'CACHED_DECODE',
                    kernel_count=len(selected),kernel_order_sha256=hashlib.sha256('\n'.join(tape).encode()).hexdigest(),
                    collective_count=len(collective),collective_order_sha256=hashlib.sha256('\n'.join(collective).encode()).hexdigest(),
                    KV_write_count=sum('reshape_and_cache' in e['name'] for e in selected),
                    attention_main_count=sum('FlashAttnFwdSm90' in e['name'] for e in selected),
                    GPU_busy_union_ms=union_ms(selected),
                    label='KERNEL_CLASSIFIED',nonoverlap_comm_ms='UNRESOLVED',cross_rank_wait_ms='UNRESOLVED'))
                tapes[(model,family,R,TP,rank,forward)]=tape
                for name,records in groups.items():
                    kernels.append(key|dict(rank=rank,forward=forward,phase='PREFILL' if forward==0 else 'CACHED_DECODE',
                        name=name,category=category(name),count=len(records),service_ms=sum(e['end_us']-e['start_us'] for e in records)/1000,
                        timing_validity=validity[directory.name]['kernel_timing']))
                for cat in sorted({category(e['name']) for e in selected}):
                    records=[e for e in selected if category(e['name'])==cat]
                    service=sum(e['end_us']-e['start_us'] for e in records)/1000
                    services.append(key|dict(rank=rank,forward=forward,phase='PREFILL' if forward==0 else 'CACHED_DECODE',
                        category=cat,count=len(records),service_ms=service,logical_depth=depth,service_ms_per_layer=service/depth,
                        label='KERNEL_CLASSIFIED_SERVICE_NOT_E2E_SHARE',timing_validity=validity[directory.name]['kernel_timing']))
    table(out/'all_config_repeats.csv',measures);table(out/'all_config_per_request.csv',requests)
    table(out/'physical_ownership.csv',owners);table(out/'kernel_and_collective_counts.csv',kernels)
    table(out/'forward_kernel_services.csv',services);table(out/'graph_kernel_signatures.csv',graph_counts)
    table(out/'outer_events.csv',outer_rows);table(out/'runtime_execution_counts.csv',runtime)
    table(out/'layer_attention_service.csv',layer_attention)
    write(out/'profile_validity.json',validity);write(out/'parity_and_kv_validation.json',parity)
    write(out/'analysis_coverage.json',coverage)
    summary=[];index={(r['model'],r['family'],r['R'],r['TP'],r['prefix'],r['request_id']):r for r in requests}
    grouped=defaultdict(list)
    for row in requests:grouped[tuple(row[k] for k in ['model','family','R','TP','prefix'])].append(row)
    comparisons=[]
    for (model,family,R,TP,prefix),rows in sorted(grouped.items()):
        key=dict(model=model,family=family,R=R,TP=TP,prefix=prefix)
        result=key|dict(n_requests=len(rows),technical_repeats_per_request=min(r['technical_repeats'] for r in rows))
        for metric in ['prefill_ms','TPOT_ms','E2E_ms','output_TPS','logical_KV_bytes_per_rank','peak_allocated_bytes']:
            stats=bootstrap([r[metric] for r in rows])
            result.update({metric:stats['mean'],metric+'_low':stats['low'],metric+'_high':stats['high']})
        summary.append(result)
        for metric in ['prefill_ms','TPOT_ms','E2E_ms']:
            values=[];interaction=[];unroll=[];unroll_interaction=[];ideal_excess=[]
            for r in rows:
                req=r['request_id'];baseline=index.get((model,family,R,1,prefix,req))
                if baseline:
                    speedup=baseline[metric]/r[metric];values.append(speedup)
                    ideal_excess.append(r[metric]-baseline[metric]/TP)
                    b1=index.get((model,family,1,1,prefix,req));bk=index.get((model,family,1,TP,prefix,req))
                    if model=='ouro' and b1 and bk:interaction.append(speedup/(b1[metric]/bk[metric]))
                if model=='ouro' and family=='tied':
                    native=index.get((model,'native',R,TP,prefix,req));n1=index.get((model,'native',R,1,prefix,req));t1=index.get((model,'tied',R,1,prefix,req))
                    if native:
                        ratio=native[metric]/r[metric];unroll.append(ratio)
                        if n1 and t1:unroll_interaction.append(ratio/(n1[metric]/t1[metric]))
            for label,data in [('TP_speedup',values),('ScalingInteraction',interaction),('NativeVsUnrolledRatio',unroll),('UnrollTPInteraction',unroll_interaction)]:
                if data:
                    stats=bootstrap(data)
                    comparisons.append(key|dict(metric=metric,comparison=label,**stats,
                        parallel_efficiency=stats['mean']/TP if label=='TP_speedup' else None))
            if ideal_excess:comparisons.append(key|dict(metric=metric,comparison='TPExcessVsIdeal',**bootstrap(ideal_excess),
                units='ms',interpretation='Diagnostic residual. NOT removable or achievable headroom.'))
    table(out/'config_summary.csv',summary);table(out/'paired_comparisons.csv',comparisons)
    for name,predicate in [('native_ouro_r_tp',lambda r:r['model']=='ouro' and r['family']=='native'),
                           ('tied_unrolled_r_tp',lambda r:r['family']=='tied'),('real_ar_qwen_tp',lambda r:r['model']!='ouro')]:
        table(out/(name+'.csv'),[r for r in summary if predicate(r)])
    signatures=[]
    for key,tape in tapes.items():
        model,family,R,TP,rank,forward=key
        if family!='tied':continue
        native=tapes.get((model,'native',R,TP,rank,forward))
        if native is None:continue
        def select(seq,kind):return [n for n in seq if category(n)==kind]
        signatures.append(dict(model=model,R=R,TP=TP,rank=rank,forward=forward,
            exact_order_equal=tape==native,native_count=len(native),tied_count=len(tape),
            collective_order_equal=select(tape,'TP_COLLECTIVE')==select(native,'TP_COLLECTIVE'),
            attention_order_equal=select(tape,'ATTENTION_WITH_KV_READ')==select(native,'ATTENTION_WITH_KV_READ'),
            GEMM_order_equal=select(tape,'QKV_FFN_OUTPUT_GEMM')==select(native,'QKV_FFN_OUTPUT_GEMM'),
            limitation='Physical kernel order. Does not identify every CUDA graph node or globally aligned rank wait.'))
    write(out/'native_tied_kernel_comparison.json',signatures)
    headline=[]
    for row in summary:
        def matching(r):return all(r[k]==row[k] for k in ['model','family','R','TP'])
        rank_rows=[r for r in owners if matching(r)];execution=[r for r in runtime if matching(r)]
        signature=[r for r in graph_counts if matching(r) and r['rank']==0]
        base=dict(row)
        base.update(revision=MODELS[row['model']]['revision'],dtype='BF16',
            physical_GPU_list=GPUS[:row['TP']],GPU_uuids=[r['GPU_uuid'] for r in rank_rows],
            actual_forward_count=execution[0]['actual_nonempty_forwards'],graph_replay_count=execution[0]['graph_replays'],
            raw_graph_node_count='UNKNOWN_IN_PRIMARY. SEE_RETAINED_GRAPH_DIAGNOSTIC',
            executed_logical_layers=rank_rows[0]['logical_depth'],physical_layers=rank_rows[0]['physical_layers'],
            cache_slots=rank_rows[0]['cache_slots'],Q_heads_local=rank_rows[0]['Q_heads_local'],KV_heads_local=rank_rows[0]['KV_heads_local'],
            weight_GiB_per_rank=[r['unique_weight_bytes']/2**30 for r in rank_rows],KV_pool_GiB_per_rank=[r['KV_pool_bytes']/2**30 for r in rank_rows],
            attention_backend='FlashAttention3',graph_policy='Native compiled FULL_AND_PIECEWISE; capture rows 1,2,4,8',
            validated_nonoverlap_comm_ms='UNRESOLVED',cross_rank_wait_ms='UNRESOLVED',
            numerical_status='PRIMARY_SELECTED_BITWISE_PASS. NATURAL_DIAGNOSTIC_SEPARATE' if row['family']=='tied' else 'NATIVE_PRETRAINED. SEE_CROSS_TP_LOGITS_AUDIT',
            measurement_validity='CLEAN_B1_WALL. SOURCE_AND_KERNEL_DIAGNOSTICS_SEPARATE')
        if row['model']=='ouro' and row['R']==4 and row['TP']==4:
            base['natural_parity_caveat']='Retained-graph natural math mismatch. See natural_discrepancy_audit.json. Do not infer universal parity.'
        base['prefill_wall_ms_per_logical_layer']=row['prefill_ms']/rank_rows[0]['logical_depth']
        base['cached_TPOT_ms_per_logical_layer']=row['TPOT_ms']/rank_rows[0]['logical_depth']
        for phase in ['PREFILL','CACHED_DECODE']:
            selected=[r for r in signature if r['phase']==phase]
            base[phase+'_collective_kernels_per_forward']=statistics.mean(r['collective_count'] for r in selected)
            base[phase+'_attention_main_kernels_per_forward']=statistics.mean(r['attention_main_count'] for r in selected)
        for metric in ['prefill_ms','TPOT_ms','E2E_ms']:
            comparison=next((r for r in comparisons if matching(r) and r['prefix']==row['prefix'] and r['metric']==metric and r['comparison']=='TP_speedup'),None)
            if comparison:
                base[metric+'_TP_speedup']=comparison['mean'];base[metric+'_TP_speedup_95CI']=[comparison['low'],comparison['high']]
                base[metric+'_parallel_efficiency']=comparison['parallel_efficiency']
        headline.append(base)
    table(out/'four_family_headline.csv',headline)
    print('Completed configurations',len(coverage),'request averages',len(requests),'parity fields',len(parity))


if __name__=='__main__':main()
