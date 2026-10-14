"""Cause validation across low-overhead profiler and source-correlated replay."""
import collections
import gzip
import statistics
from poc_dllm_multigpu_e2e_atlas.contract import *
from poc_dllm_multigpu_e2e_atlas.intervals import kernel_partition,union_ms
from poc_dllm_multigpu_e2e_atlas.scripts.parse_timeline import classify,table


def load(path):
    with gzip.open(path,'rt') as stream:return json.load(stream)


def main():
    raw=ROOT/'results/raw/torch_profile_p16384_v1'
    all_kernels=[];services=[];partitions=[];validity=[]
    for rank in range(4):
        prof=load(raw/f'coding_00.p16384.profile.0.rank{rank}.json.gz')
        clean=[load(raw/f'coding_00.p16384.clean.{rep}.rank{rank}.json.gz') for rep in range(3)]
        assert all(c['output_ids']==prof['output_ids'] and c['NFE']==prof['NFE'] for c in clean)
        overhead=prof['wall_ms']/statistics.mean(c['wall_ms'] for c in clean)-1
        validity.append(dict(rank=rank,wall_ms=prof['wall_ms'],overhead=overhead,output_NFE_parity=True,
            valid_bounded=overhead<=.10,scope='One calibration request. No request-population CI.'))
        events=load(raw/f'torch_events.rank{rank}.json.gz')
        kernels=[]
        for event in events:
            if 'CUDA' not in event['device_type']:continue
            category=classify(event['name'])
            if event['name']=='memcpy128':category='FULL_BACKING_STORAGE_COPY_CONFIRMED_BY_SEPARATE_SOURCE_TRACE'
            kernels.append(dict(rank=rank,name=event['name'],start=event['start_us']/1000,
                end=event['end_us']/1000,category=category))
        all_kernels.extend(kernels)
        grouped=collections.defaultdict(list)
        for k in kernels:grouped[k['category']].append(k)
        for category,values in grouped.items():services.append(dict(rank=rank,category=category,count=len(values),
            service_ms=sum(k['end']-k['start'] for k in values),
            union_ms=union_ms([(k['start'],k['end']) for k in values]),scope='GPU_SERVICE_NOT_E2E_PERCENT'))
        for part in kernel_partition(kernels,min(k['start'] for k in kernels),max(k['end'] for k in kernels)):
            partitions.append(dict(rank=rank,**part,scope='EXCLUSIVE_RANK_ACTIVITY_BOUND; full graph dependency attribution PARTIAL'))
    source=ROOT/'results/raw/parsed_node_p16384_source_v1'
    copies=load(source/'copies.json.gz')
    proof=[]
    for rank in range(4):
        diag=json.loads((ROOT/f'results/raw/nsys_node_p16384_source_v1/eager_source_diagnostic.rank{rank}.json').read_text())
        actual=[r for r in copies if r['rank']==rank and not r['diagnostic'] and
            r['scope'].startswith('ATLAS|graph_replay|') and r['bytes']==335544320]
        source_ops=[r for r in copies if r['rank']==rank and r['diagnostic'] and
            'KV_SLICE_SCATTER' in r['scope'] and r['bytes']==335544320]
        assert len(actual)==560 and len(source_ops)==40
        assert len(diag['tensor_operations'])==40 and diag['logit_max_abs']==0
        assert all(o['full_tensor_bytes']==8388608 and o['source_bytes']==8192 for o in diag['tensor_operations'])
        observed=len({r['correlation'] for r in actual})
        proof.append(dict(rank=rank,logical_KV_view_bytes=8388608,actual_full_storage_copy_bytes=335544320,
            physical_copy_inflation=40,observed_regular_or_refresh_graphs=observed,
            native_graph_full_storage_copy_count=len(actual),source_diagnostic_copy_count=len(source_ops),
            native_graph_D2D_bytes=sum(r['bytes'] for r in actual),
            source_same_input_eager_logits_exact=True,
            source='native LLaDA2Attention.forward -> Tensor.slice_scatter on a view of monolithic graph KV storage',
            all_to_all=False))
    table(ROOT/'results/gpu_kernel_service.csv',services)
    table(ROOT/'results/copy_and_memory_classification.csv',proof)
    table(ROOT/'results/profile_exclusive_rank_activity.csv',partitions)
    with gzip.open(ROOT/'results/torch_kernel_occurrences.json.gz','wt') as stream:json.dump(all_kernels,stream)
    write(ROOT/'manifests/profile_validity.json',dict(torch_profile=validity,
        source_trace='Heavy node/eager diagnostic only. Its duration cannot be a headline component cost.',
        actual_storage_copy='CUPTI D2D byte counts and source NVTX agree. Not inferred from memcpy128 name alone.',
        stream_critical_path='PARTIAL. Exclusive activity is a bound. Only a paired exact intervention establishes removable E2E cost.'))
    write(ROOT/'results/repeated_cost_suspects.json',dict(
        source_proof=proof,profile_validity=validity,
        leading_target='Oversized backing-storage clone in out-of-place KV slice_scatter',
        preserved_work='Fresh QKV, native cache content/mask, out-of-place cache ownership, original sampler and weights',
        not_target='Prefix projection caching already exists. No EP A2A. No stale attention reuse.',
        next_gate='One exact compact out-of-place scatter control, after architecture screening. No new engine.'))
    print('Physical40x copy footprint confirmed across4 ranks; low-overhead profiler overhead',
        [round(r['overhead']*100,2) for r in validity])
    for rank in range(4):
        copy=sum(p['end']-p['start'] for p in partitions if p['rank']==rank and p['category'].startswith('FULL_BACKING'))
        print('rank',rank,'exclusive copy activity ms',copy,'not yet measured E2E saving')


if __name__=='__main__':main()
