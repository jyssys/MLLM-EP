"""Summarize the separate retained-graph, natural, and observer lanes."""
from collections import defaultdict
import gzip
import statistics
from poc_looped_recurrence_tp_4models.contract import *
from poc_looped_recurrence_tp_4models.analysis import read,table,bootstrap,category,union_ms


def main():
    root=ROOT/'results/diagnostics';graphs=[];hooks=[];natural=[];paired=[];all_natural={}
    tapes={};services=[];CPU=[];rechecks={}
    for directory in sorted(root.glob('*')):
        if not (directory/'status.json').exists():continue
        status=read(directory/'status.json')
        if not status.get('success'):continue
        actual_R=read(directory/'engine_contract.json').get('hf_overrides',{}).get('total_ut_steps',1)
        key={k:status[k] for k in ['model','family','tp']};key['r']=actual_R
        for rank,rows in enumerate(read(directory/'graphs.json')):
            for row in rows:graphs.append(key|dict(rank=rank,**row))
        data=read(directory/'teacher_hook_overhead.json');by_repeat=defaultdict(dict)
        for row in data['pairs']:by_repeat[row['repeat']][row['arm']]=row
        overhead=[(r['forced_copy']['wall_ms']/r['native_sample']['wall_ms']-1)*100 for r in by_repeat.values()]
        hooks.append(key|dict(n_independent_requests=1,technical_pairs=len(overhead),
            mean_overhead_percent=statistics.mean(overhead),median_overhead_percent=statistics.median(overhead),
            minimum_percent=min(overhead),maximum_percent=max(overhead),scope='Same trajectory, retained-graph diagnostic. Not a primary latency correction.'))
        rows=read(directory/'natural_generation.json')['results'];all_natural[directory.name]=rows
        profile=read(directory/'P1024_kernel_profile.json')
        validity='UNVALIDATED' if profile['observer_percent']>10 else 'DIAGNOSTIC_SINGLE_PROFILE'
        for path in sorted(directory.glob('torch_events.rank*.json.gz')):
            rank=int(path.name.split('rank')[1].split('.')[0])
            with gzip.open(path,'rt') as stream:events=json.load(stream)
            gpu=sorted([e for e in events if 'CUDA' in e['device_type']],key=lambda e:e['start_us'])
            markers=[e['start_us'] for e in gpu if e['name']=='_compute_slot_mappings_kernel']
            assert len(markers)==4
            for forward,start in enumerate(markers):
                stop=markers[forward+1] if forward<3 else float('inf')
                selected=[e for e in gpu if start<=e['start_us']<stop]
                tapes[(directory.name,rank,forward)]=[e['name'] for e in selected if 'memcpy' not in e['name'].lower() and 'memset' not in e['name'].lower()]
                services.append(key|dict(config=directory.name,rank=rank,forward=forward,
                    observer_percent=profile['observer_percent'],validity=validity,GPU_busy_union_ms=union_ms(selected),
                    execution_span_ms=(max(e['end_us'] for e in selected)-min(e['start_us'] for e in selected))/1000,
                    **{cat:sum(e['end_us']-e['start_us'] for e in selected if category(e['name'])==cat)/1000 for cat in set(category(e['name']) for e in selected)}))
            for name in ['cudaGraphLaunch','cudaLaunchKernel','cuLaunchKernel','cuLaunchKernelEx','cudaStreamWaitEvent','cudaEventSynchronize','TorchDynamo Cache Lookup']:
                selected=[e for e in events if 'CPU' in e['device_type'] and name in e['name']]
                CPU.append(key|dict(config=directory.name,rank=rank,name=name,count=len(selected),
                    nested_CPU_service_ms=sum(e['end_us']-e['start_us'] for e in selected)/1000,
                    scope='Whole diagnostic request plus RPC setup. CPU service can nest or overlap GPU work. Not critical-path waiting.'))
        repeat_file=directory/'source_grid_repeat.json'
        if repeat_file.exists():
            grouped_repeat=defaultdict(list)
            for row in read(repeat_file)['results']:grouped_repeat[row['request_id']].append(row)
            rechecks[directory.name]={request:dict(E2E_ms=statistics.mean(r['wall_ms'] for r in records),
                prefill_ms=statistics.mean((r['metrics']['first_token_ts']-r['metrics']['scheduled_ts'])*1000 for r in records),
                TPOT_ms=statistics.mean((r['metrics']['last_token_ts']-r['metrics']['first_token_ts'])*1000/63 for r in records))
                for request,records in grouped_repeat.items()}
        grouped=defaultdict(list)
        for row in rows:grouped[row['request_id']].append(row)
        for request,records in grouped.items():
            natural.append(key|dict(request_id=request,output_tokens=len(records[0]['ids']),finish_reason=records[0]['finish_reason'],
                repeat_token_parity=all(r['ids']==records[0]['ids'] for r in records),
                scope='Bounded natural sanity. Not benchmark accuracy. No code execution.'))
    for name,rows in all_natural.items():
        if '_tied_' not in name:continue
        reference=all_natural.get(name.replace('_tied_','_native_'))
        if reference is None:continue
        index={(r['request_id'],r['repeat']):r for r in reference}
        for row in rows:
            other=index[(row['request_id'],row['repeat'])]
            paired.append(dict(config=name,request_id=row['request_id'],repeat=row['repeat'],
                same_token_IDs=row['ids']==other['ids'],same_stop_reason=row['finish_reason']==other['finish_reason'],
                same_prompt_IDs=row['prompt_ids']==other['prompt_ids'],
                native_output_tokens=len(other['ids']),tied_output_tokens=len(row['ids']),
                first_different_index=next((i for i,(a,b) in enumerate(zip(other['ids'],row['ids'])) if a!=b),
                    None if len(other['ids'])==len(row['ids']) else min(len(other['ids']),len(row['ids'])))))
    table(ROOT/'results/retained_graph_nodes.csv',graphs);table(ROOT/'results/teacher_hook_overhead.csv',hooks)
    table(ROOT/'results/natural_sanity.csv',natural);write(ROOT/'results/natural_native_tied_parity.json',paired)
    comparisons=[]
    for row in graphs:
        if row['family']!='tied':continue
        reference=next((r for r in graphs if r['model']==row['model'] and r['r']==row['r'] and r['family']=='native' and
            r['tp']==row['tp'] and r['rank']==row['rank'] and r['descriptor']==row['descriptor']),None)
        if reference:
            assert reference['r']==row['r'],'Do not compare different recurrent depths'
            comparisons.append(dict(R=row['r'],TP=row['tp'],rank=row['rank'],descriptor=row['descriptor'],
                native_nodes=reference['node_count'],tied_nodes=row['node_count'],
                native_edges=reference['edge_count'],tied_edges=row['edge_count'],
                same_node_counts=row['node_count']==reference['node_count'],same_edge_counts=row['edge_count']==reference['edge_count'],
                same_type_counts=row['node_type_counts']==reference['node_type_counts'],
                same_dependency_type_fingerprint=None if 'UNRESOLVED' in [row['dependency_type_fingerprint_8_rounds'],reference['dependency_type_fingerprint_8_rounds']]
                    else row['dependency_type_fingerprint_8_rounds']==reference['dependency_type_fingerprint_8_rounds'],
                scope='Retained graph diagnostic. Matching fingerprints are not a formal graph-isomorphism proof or a kernel-parameter comparison.'))
    write(ROOT/'results/retained_graph_comparison.json',comparisons)
    kernel_checks=[];repeat_checks=[]
    for (name,rank,forward),tape in tapes.items():
        if '_tied_' not in name:continue
        reference=tapes.get((name.replace('_tied_','_native_'),rank,forward))
        if reference is not None:kernel_checks.append(dict(config=name,rank=rank,forward=forward,
            exact_model_kernel_order=tape==reference,native_count=len(reference),tied_count=len(tape)))
    for name,values in rechecks.items():
        if '_tied_' not in name:continue
        reference=rechecks.get(name.replace('_tied_','_native_'))
        if not reference:continue
        for metric in ['E2E_ms','prefill_ms','TPOT_ms']:
            data=[reference[request][metric]/row[metric] for request,row in values.items()]
            repeat_checks.append(dict(config=name,metric=metric,comparison='Native / Tied',**bootstrap(data),
                scope='Post-hoc independent-session recheck of calibration requests. Not untouched held-out. Retained-graph lane.'))
    write(ROOT/'results/P1024_model_kernel_checks.json',kernel_checks)
    write(ROOT/'results/independent_session_rechecks.json',repeat_checks)
    table(ROOT/'results/P1024_GPU_services.csv',services);table(ROOT/'results/P1024_CPU_services.csv',CPU)
    print('Graph records',len(graphs),'natural probes',len(natural),'native/tied natural pairs',len(paired))


if __name__=='__main__':main()
