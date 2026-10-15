"""Aggregate native measurements. Keep pool capacity separate from active KV."""
import collections
import statistics
from poc_looped_multigpu_kv_placement.contract import *
from poc_looped_multigpu_kv_placement.analysis import bootstrap,kv_bytes_per_token,table


def main():
    requests=[];batches=[];memory=[];proof=[]
    for directory in sorted((ROOT/'results/raw').glob('ouro_tp*_native*')):
        ownership=directory/'loaded_ownership.json'
        if not ownership.exists():continue
        owned=json.loads(ownership.read_text());tp=owned[0]['world']
        for rank in owned:
            slots=rank['cache_slots'];assert slots==192 and rank['physical_layers']==48
            assert all(g['KV_heads']==16//tp and g['Q_heads']==16//tp and g['loop_cache_modules']==4 for g in rank['geometry'])
            logical=kv_bytes_per_token(slots,16,128,tp=tp)
            physical=rank['unique_KV_pool_storage_bytes']
            proof.append(dict(source=directory.name,rank=rank['rank'],GPU_uuid=rank['GPU_uuid'],tp=tp,
                cache_slots=slots,physical_layers=48,KV_heads_per_slot=16//tp,logical_KV_bytes_per_token=logical,
                KV_pool_storage_bytes=physical,KV_pool_blocks=rank['KV_pool_blocks'],page_tokens=16,
                pool_token_capacity=rank['KV_pool_blocks']*16,
                measured_parameter_storage=rank['unique_parameter_storage_bytes'],allocated=rank['memory_allocated'],
                reserved=rank['memory_reserved'],
                other_allocated=rank['memory_allocated']-physical-rank['unique_parameter_storage_bytes'],
                status='ACTUAL_LOADED_TENSORS. Logical filled KV and allocated pool are different accounts.'))
        for path in sorted(directory.glob('P*.repeat*.json')):
            if path.name.endswith('.memory.json'):continue
            value=json.loads(path.read_text());p=value['prefix'];concurrency=value['concurrency']
            pool_capacity=min(rank['KV_pool_blocks']*16 for rank in owned)
            cap=pool_capacity//(((p+64+15)//16)*16)
            batches.append(dict(source=directory.name,label=path.stem,tp=tp,prefix=p,concurrency=concurrency,
                repeat=value['repeat'],requests=len(value['requests']),wall_ms=value['wall_ms'],
                aggregate_TPS=64*len(value['requests'])/(value['wall_ms']/1000),
                KV_page_derived_resident_limit=cap,offered_exceeds_KV_capacity=concurrency>cap,
                note='Offered concurrency is not proof of simultaneous residency. Queueing/preemption can serialize excess requests.'))
            for row in value['requests']:
                stats=row['metrics'];assert stats is not None
                requests.append(dict(source=directory.name,tp=tp,prefix=p,concurrency=concurrency,repeat=value['repeat'],
                    request_id=row['request_id'],domain=row['domain'],tokens=len(row['output_ids']),
                    batch_wall_ms=value['wall_ms'],
                    engine_prefill_ms=(stats['first_token_ts']-stats['scheduled_ts'])*1000,
                    engine_TPOT_ms=(stats['last_token_ts']-stats['first_token_ts'])*1000/(len(row['output_ids'])-1),
                    engine_request_ms=(stats['last_token_ts']-stats['queued_ts'])*1000,
                    prefix_cache_hit_tokens=row['num_cached_tokens'],finish_reason=row['finish_reason']))
            ram_path=path.with_name(path.stem+'.memory.json')
            if ram_path.exists():
                for rank,usage in enumerate(json.loads(ram_path.read_text())):
                    memory.append(dict(source=directory.name,label=path.stem,tp=tp,prefix=p,concurrency=concurrency,
                        rank=rank,logical_filled_KV_bytes_per_request=kv_bytes_per_token(192,16,128,tp=tp)*(p+63),
                        **usage,KV_pool_storage_bytes=owned[rank]['unique_KV_pool_storage_bytes'],
                        weights=owned[rank]['unique_parameter_storage_bytes']))
    table(ROOT/'results/request_metrics.csv',requests);table(ROOT/'results/capacity_frontier.csv',batches)
    table(ROOT/'results/memory_per_rank.csv',memory);write(ROOT/'results/tp_ownership_proof.json',proof)
    groups={}
    for tp,prefix in sorted({(r['tp'],r['prefix']) for r in requests if r['concurrency']==1}):
        selected=[r for r in requests if (r['tp'],r['prefix'],r['concurrency'])==(tp,prefix,1)]
        fields={}
        for field in ['batch_wall_ms','engine_prefill_ms','engine_TPOT_ms','engine_request_ms']:
            by_request=collections.defaultdict(list)
            for row in selected:by_request[row['request_id']].append(row[field])
            fields[field]=bootstrap([statistics.mean(v) for v in by_request.values()])
        groups[f'TP{tp}_P{prefix}']=fields
    write(ROOT/'results/native_summary.json',dict(groups=groups,
        clean_wall='Client monotonic generate call. Loaded/warmed engine. Input IDs already tokenized. Native output serialization included.',
        engine_intervals='Native scheduler timestamps, not kernel service. Decode TPOT excludes first-token prefill.',
        statistics='2000 request-bootstrap resamples. Average technical repeats before resampling.',
        partial='A running campaign may have incomplete groups. Final gates require completion records.'))
    print('Derived native request, capacity, HBM and ownership tables. No speedup from bytes.')


if __name__=='__main__':main()
