"""Check actual ownership, execution, parity, and source-policy consistency."""
from collections import defaultdict
import numpy as np
from poc_looped_recurrence_tp_4models.contract import *
from poc_looped_recurrence_tp_4models.analysis import read,table


def main():
    raw=ROOT/'results/raw';checks=[];cross_tp=[];hot_hashes=defaultdict(set)
    for directory in sorted(raw.glob('*_v1')):
        if 'smoke' in directory.name or not (directory/'status.json').exists():continue
        status=read(directory/'status.json')
        if not status.get('success'):continue
        model=status['model'];family=status['family'];R=status['r'];TP=status['tp'];owned=read(directory/'ownership.json')
        depth=48*R if model=='ouro' else 28 if model=='qwen3' else 36
        rows=dict(config=directory.name,loaded_rank_count=len(owned),expected_rank_count=TP,
            correct_UUID_mapping=all(r['GPU_uuid'].removeprefix('GPU-')==UUIDS[i] for i,r in enumerate(owned)),
            correct_cache_slots=all(r['cache_slots']==depth for r in owned),native_V2=all(
                r['model_runner']=='vllm.v1.worker.gpu.model_runner' for r in owned),
            all_BF16_weights=all(p['dtype']=='torch.bfloat16' for r in owned for p in r['parameters'].values()
                if p['logical_bytes']>16),no_EP=all('enable_expert_parallel=False' in r['actual_parallel'] for r in owned))
        if model=='ouro':
            rows['tape_has_correct_depth']=all(len(r['execution_tape'])==depth for r in owned)
            rows['shared_parameter_storage']=all(len({x['qkv_storage_ptr'] for x in r['execution_tape'] if x['physical_layer']==layer})==1
                for r in owned for layer in range(48))
            rows['independent_KV_slot_names']=all(len({x['cache_prefix'] for x in r['execution_tape']})==depth for r in owned)
            rows['tape_slots_match_loaded_cache']=all({x['cache_prefix'] for x in r['execution_tape']}=={x['name'] for x in r['cache']} for r in owned)
            rows['distinct_logical_KV_allocations']=all(len({x['storage_ptr'] for x in r['cache']})==depth for r in owned)
        if model=='qwen25' and TP==4:
            rows['KV_replication_pairs_all_layers']=all(owned[0]['geometry'][layer][name]==owned[1]['geometry'][layer][name]
                and owned[2]['geometry'][layer][name]==owned[3]['geometry'][layer][name]
                and owned[0]['geometry'][layer][name]!=owned[2]['geometry'][layer][name]
                for layer in range(36) for name in ['K_weight_sha256','V_weight_sha256'])
            arrays=[]
            for rank in range(4):
                with np.load(directory/f'parity_P16384.rank{rank}.npz') as stream:arrays.append(stream['KV'])
            rows['KV_sample_replication_pairs']=bool(np.array_equal(arrays[0],arrays[1]) and np.array_equal(arrays[2],arrays[3]))
        if family=='tied':
            rows['exact_logits_hidden_KV']=all(c['bitwise_equal'] for p in directory.glob('parity_check_*.json') for c in read(p))
            native=read((directory.parent/directory.name.replace('_tied_','_native_'))/'ownership.json')
            rows['same_weight_storage_sizes_as_native']=all(r['unique_parameter_storage_bytes']==n['unique_parameter_storage_bytes'] for r,n in zip(owned,native))
        outer=read(directory/'outer_and_counts.json')
        rows['actual_64_forwards']=all(sum(e['category']=='execute_model' and (e['scheduled_rows'] or 0)>0 for e in events)==64 for events in outer['outer'])
        rows['actual_63_replays']=all(r['graph_replays']==63 for r in outer['counts'])
        rows['actual_64_sampler_calls']=all(r['sampling_calls']==64 for r in outer['counts'])
        contract=read(directory/'run_contract.json')
        def walk(value):
            if isinstance(value,dict):
                for k,v in value.items():
                    if k.endswith(('worker_probe.py','scripts/run_config.py','tied_tape.py')) and isinstance(v,str):hot_hashes[k].add(v)
                    walk(v)
            elif isinstance(value,list):
                for item in value:walk(item)
        walk(contract)
        checks.append(rows)
        if TP>1 and family=='native':
            reference=raw/directory.name.replace(f'_TP{TP}_','_TP1_')
            if reference.exists():
                for prefix in [1024,16384]:
                    a_path=reference/f'parity_P{prefix}.rank0.npz';b_path=directory/f'parity_P{prefix}.rank0.npz'
                    if not a_path.exists() or not b_path.exists():continue
                    with np.load(a_path) as stream:a=stream['logits']
                    with np.load(b_path) as stream:b=stream['logits']
                    differences=np.flatnonzero(np.argmax(a,axis=-1)!=np.argmax(b,axis=-1))
                    cross_tp.append(dict(model=model,R=R,TP=TP,prefix=prefix,first_greedy_difference=int(differences[0]) if len(differences) else None,
                        greedy_prediction_equal=not len(differences),bitwise_logits_equal=bool(np.array_equal(a,b)),
                        max_abs=float(np.abs(a-b).max()),relative_L2=float(np.linalg.norm((a-b).ravel())/max(np.linalg.norm(a.ravel()),1e-30)),
                        scope='Four unmodified logit vectors under identical teacher prefix. Not natural long-generation equivalence.'))
    table(ROOT/'results/ownership_and_execution_checks.csv',checks)
    write(ROOT/'results/cross_TP_logits.json',cross_tp)
    write(ROOT/'results/hot_source_hashes.json',{k:sorted(v) for k,v in hot_hashes.items()})
    bool_checks=[value for row in checks for value in row.values() if isinstance(value,bool)]
    print('Configurations',len(checks),'boolean checks',len(bool_checks),'failed',sum(not x for x in bool_checks))


if __name__=='__main__':main()
