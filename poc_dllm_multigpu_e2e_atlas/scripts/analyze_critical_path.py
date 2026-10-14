"""New request statistics and actual every-replay outer budgets."""
import collections
import gzip
import statistics
import numpy as np
from poc_dllm_multigpu_e2e_atlas.contract import *
from poc_dllm_multigpu_e2e_atlas.intervals import exclusive_budget
from poc_dllm_multigpu_e2e_atlas.scripts.parse_timeline import table
from poc_dllm_ep_comm.metrics import bootstrap_mean


def load(path):
    with gzip.open(path,'rt') as stream:return json.load(stream)


def main():
    requests=[];refinements=[];blocks=[];intervals=[];phases=[];checks=[];summary={}
    for prefix in PREFIXES:
        if prefix==512:directory=ROOT/'results/raw/native_p512_v2'
        else:
            directory=ROOT/f'results/raw/native_p{prefix}_v1'
            if not (directory/'status.json').exists():directory=ROOT/f'results/raw/native_p{prefix}_eager_fallback_v1'
        if not (directory/'status.json').exists():continue
        for row in rows():
            rid=row['request_id'];data={mode:[] for mode in ['clean','coarse','parity']}
            for mode in data:
                for repeat in range(1 if mode=='parity' else 3):
                    data[mode].append([load(directory/f'{rid}.p{prefix}.{mode}.{repeat}.rank{rank}.json.gz') for rank in range(4)])
            truth=data['clean'][0][0]
            for mode,runs in data.items():
                for repeat,run in enumerate(runs):
                    for rank,record in enumerate(run):
                        okay=record['output_ids']==truth['output_ids'] and record['NFE']==truth['NFE']
                        if mode!='clean':okay=okay and record['observation']['updates']==data['parity'][0][0]['observation']['updates']
                        checks.append(dict(request_id=rid,prefix=prefix,mode=mode,repeat=repeat,rank=rank,pass_all=okay))
                        assert okay,(rid,prefix,mode,repeat,rank)
            rank=int(np.argmax([sum(run[r]['wall_ms'] for run in data['coarse']) for r in range(4)]))
            clean=statistics.mean(max(r['wall_ms'] for r in run) for run in data['clean'])
            coarse=statistics.mean(max(r['wall_ms'] for r in run) for run in data['coarse'])
            features=collections.defaultdict(list)
            for repeat,run in enumerate(data['coarse']):
                record=run[rank];spans=record['observation']['spans'];by_id={s['id']:s for s in spans}
                root=next(s for s in spans if s['kind']=='request_gpu')
                host_parts=exclusive_budget(spans,record['start_host_ns'],record['completion_host_ns'])
                grouped=collections.defaultdict(float)
                for part in host_parts:grouped[part['kind']]+=(part['end']-part['start'])/1e6
                for kind,ms in grouped.items():phases.append(dict(request_id=rid,prefix=prefix,rank=rank,repeat=repeat,
                    category=kind,exclusive_host_ms=ms,denominator_ms=record['wall_ms'],
                    interpretation='Disjoint host-wall scopes, not GPU compute or a full dependency critical path'))
                assert abs(sum(grouped.values())-record['wall_ms'])<1e-6
                for kind in ['prefill','refinement','commit_refresh','graph_input_staging','graph_replay','sampling_acceptance','KV_range_update','KV_extend_cache']:
                    chosen=[s for s in spans if s['kind']==kind]
                    features[kind+'_GPU_interval_ms'].append(sum(s['GPU_interval_ms'] for s in chosen))
                    features[kind+'_host_ms'].append(sum(s['host_interval_ms'] for s in chosen))
                    features[kind+'_count'].append(len(chosen))
                features['GPU_request_interval_ms'].append(root['GPU_interval_ms'])
                features['outer_unknown_GPU_ms'].append(root['GPU_interval_ms']-sum(s['GPU_interval_ms'] for s in spans
                    if s['kind'] in ['prefill','refinement','commit_refresh','output_finalization','KV_range_update','KV_extend_cache']
                    and not any(by_id[p]['kind'] in ['prefill','refinement','commit_refresh'] for p in ancestors(s,by_id))))
                # Every forward is causally necessary. This residual is an optimistic
                # ceiling for ALL work outside graph execution, not a measured saving.
                graph=sum(s['GPU_interval_ms'] for s in spans if s['kind']=='graph_replay')
                features['outside_graph_optimistic_fraction'].append(max(0,record['wall_ms']-graph)/record['wall_ms'])
                for span in spans:
                    if span['kind']=='block':
                        block_index=sum(s['kind']=='block' and s['id']<span['id'] for s in spans)
                        blocks.append(dict(request_id=rid,prefix=prefix,rank=rank,repeat=repeat,block_index=block_index,**span))
                    if span['kind'] not in ['refinement','commit_refresh']:continue
                    children=[s for s in spans if span['id'] in ancestors(s,by_id)]
                    iteration=span['iteration'];update=next(u for u in record['observation']['updates'] if u['iteration']==iteration)
                    live_after=sum(x==156895 for x in update['ids'][0])
                    previous=[u for u in record['observation']['updates'] if u['iteration']['block']==iteration['block'] and
                        u['iteration']['step']==iteration['step']-1]
                    live_before=sum(x==156895 for x in previous[0]['ids'][0]) if previous else 32
                    datum=dict(request_id=rid,prefix=prefix,rank=rank,repeat=repeat,block=iteration['block'],step=iteration['step'],
                        cross_refresh=iteration['cross'],query_rows=iteration['end']-iteration['start'],physical_keys=2*prefix,
                        GPU_start_ms=span['GPU_start_ms'],GPU_end_ms=span['GPU_end_ms'],
                        GPU_interval_ms=span['GPU_interval_ms'],host_interval_ms=span['host_interval_ms'],
                        live_before=live_before,live_after=live_after,newly_accepted=live_before-live_after,
                        actual_EP_A2A_count=0,attention_critical_ms='UNRESOLVED',MoE_critical_ms='UNRESOLVED',
                        TP_critical_ms='UNRESOLVED',GPU_idle_critical_ms='UNRESOLVED',
                        validity='DIRECT_ALL_OCCURRENCE_OUTER_INTERVAL; operator critical attribution requires profile/dependency evidence')
                    for kind in ['graph_replay','graph_input_staging','sampling_acceptance']:
                        datum[kind+'_ms']=sum(s['GPU_interval_ms'] for s in children if s['kind']==kind)
                    refinements.append(datum)
                for part in exclusive_budget(spans,root['GPU_start_ms'],root['GPU_end_ms'],'GPU_start_ms','GPU_end_ms'):
                    intervals.append(dict(request_id=rid,prefix=prefix,rank=rank,repeat=repeat,**part,
                        validity='DIRECT_RANK_LOCAL_SCOPE_PARTITION_NOT_FULL_CROSS_RANK_CRITICAL_DAG'))
            result=dict(request_id=rid,domain=row['domain'],prefix=prefix,critical_rank=rank,
                clean_E2E_ms=clean,coarse_E2E_ms=coarse,observer_overhead=coarse/clean-1,NFE=truth['NFE'],
                output_tokens=truth['output_tokens'],TPF=truth['output_tokens']/truth['NFE'],
                cold_TPS=truth['output_tokens']/(clean/1000),eager_prefill=truth['eager_prefill'],
                peak_allocated=max(r['peak_allocated'] for run in data['clean'] for r in run),
                peak_reserved=max(r['peak_reserved'] for run in data['clean'] for r in run),
                source=str(directory.relative_to(ROOT)),**{k:statistics.mean(v) for k,v in features.items()})
            result['coarse_block_span_ms']=statistics.mean(sum(s['GPU_interval_ms'] for s in run[rank]['observation']['spans'] if s['kind']=='block') for run in data['coarse'])
            requests.append(result)
        chosen=[r for r in requests if r['prefix']==prefix]
        fields=[k for k,v in chosen[0].items() if isinstance(v,(float,int)) and k not in ['prefix','critical_rank','eager_prefill']]
        summary[str(prefix)]={k:bootstrap_mean([r[k] for r in chosen]) for k in fields}
        summary[str(prefix)]['N']=len(chosen)
        summary[str(prefix)]['observer_valid']=summary[str(prefix)]['observer_overhead']['ci95'][1]<=.10
        for r in requests:
            if r['prefix']==prefix:r['coarse_validity']='VALIDATED_OUTER_ONLY' if summary[str(prefix)]['observer_valid'] else 'UNTRUSTED_OBSERVER_OVERHEAD'
    for name,data in [('per_request',requests),('per_refinement',refinements),('per_block',blocks),
        ('critical_path_intervals',intervals),('e2e_phase_budget',phases)]:table(ROOT/f'results/{name}.csv',data)
    write(ROOT/'results/bootstrap_cis.json',dict(prefixes=summary,bootstrap='2000 request resamples',
        no_old_latency_pooling=True,no_last_replay_multiplication=True,
        critical_path_status='PARTIAL; actual whole-iteration envelopes. Fine graph dependencies remain unresolved.'))
    write(ROOT/'manifests/timing_validity.json',dict(checks=checks,all_parity=all(r['pass_all'] for r in checks),
        exact_fields=['output_ids','NFE','native token-update timeline in coarse vs separate parity-only run'],
        clean_hook_bypass=True,every_replay_recorded=True,observed_prefixes=list(summary),
        full_fine_attribution='UNRESOLVED until physical trace and causal validation',benchmark_accuracy='NOT_RUN'))
    print('New native request-length episodes',len(requests),'all parity checks',len(checks),'PASS')
    for prefix,v in summary.items():print(prefix,'clean',round(v['clean_E2E_ms']['mean'],3),
        'overhead%',round(v['observer_overhead']['mean']*100,2),'outside-graph upper%',
        round(v['outside_graph_optimistic_fraction']['mean']*100,2),v['observer_valid'])


def ancestors(span,by_id):
    result=[];parent=span['parent']
    while parent is not None:result.append(parent);parent=by_id[parent]['parent']
    return result


if __name__=='__main__':main()
