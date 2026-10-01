"""Aggregate local CUDA spans and logical route counters without inventing clocks."""
from collections import defaultdict
import numpy as np
from poc_native_lp_ep4.scripts.analyze import bootstrap


def component_waves(rows):
    grouped=defaultdict(list)
    for row in rows:grouped[(row['run'],row['wave'])].append(row)
    result=[]
    for (run,wave),events in grouped.items():
        active=[e for e in events if e['prefill_active']]
        if not active:continue
        by_rank={str(r):{k:0. for k in ('dispatch_ms','expert_execution_ms','combine_ms','moe_ms')} for r in range(4)}
        complete=all(e['ranks']==4 and len(e['rank_durations'])==4 for e in active)
        for e in active:
            for rank,times in e['rank_durations'].items():
                for k,v in times.items():by_rank[rank][k]+=v
        critical=max(by_rank,key=lambda r:by_rank[r]['moe_ms'])
        stages=defaultdict(list)
        for e in active:stages[e['layer']].append(e)
        hist_valid=all(e['injected_histogram'] is not None for e in active)
        numerator=denominator=paths=0
        traffic=np.zeros((4,4),dtype=np.int64)
        if hist_valid:
            for layer,es in stages.items():
                hs=np.asarray([e['injected_histogram'] for e in es])
                numerator+=sum(np.count_nonzero(h) for h in hs)
                denominator+=np.count_nonzero(hs.sum(0))
                paths+=int(hs.sum())
                for e in es:
                    if e.get('logical_injected_dispatch_matrix') is not None:
                        traffic+=np.asarray(e['logical_injected_dispatch_matrix'])
        expert_bytes=next((e['loaded_expert_weight_bytes_per_expert'] for e in active if e['loaded_expert_weight_bytes_per_expert']),None)
        first=active[0]
        t=by_rank[critical]
        mean_rank_moe=sum(r['moe_ms'] for r in by_rank.values())/4
        iteration_dispatch={}
        for event in active:
            key=str(event['iteration'])
            iteration_dispatch[key]=iteration_dispatch.get(key,0)+event['rank_durations'][critical]['dispatch_ms']
        result.append({'run':run,'wave':wave,'arm':first['arm'],'modality':first['modality'],
            'repeat':first['repeat'],'complete_four_rank_frames':complete,
            'prefill_active_layer_invocations':len(active),'unique_layers':len(stages),
            'rank_local_cumulative_ms':by_rank,'critical_rank_by_cumulative_moe':int(critical),
            'dispatch_total_ms':t['dispatch_ms'],'expert_total_ms':t['expert_execution_ms'],
            'max_rank_expert_total_ms':max(r['expert_execution_ms'] for r in by_rank.values()),
            'combine_total_ms':t['combine_ms'],'moe_total_ms':t['moe_ms'],
            'rank_cumulative_moe_max_mean':None if mean_rank_moe==0 else t['moe_ms']/mean_rank_moe,
            'critical_rank_dispatch_sum_by_active_iteration_ms':iteration_dispatch,
            'max_iteration_accumulated_dispatch_ms':max(iteration_dispatch.values()),
            'communication_share':None if t['moe_ms']==0 else (t['dispatch_ms']+t['combine_ms'])/t['moe_ms'],
            'max_dispatch_quantum_ms':max(e['dispatch_ms'] for e in active),
            'expert_reload_amp':numerator/denominator if denominator else None,
            'unique_prompt_expert_layer_count':denominator if denominator else None,
            'repeated_expert_layer_count':numerator if denominator else None,
            'expert_weight_bytes_proxy':numerator*expert_bytes if denominator and expert_bytes else None,
            'routed_injected_paths':paths if hist_valid else None,
            'logical_dispatch_matrix':traffic.tolist() if hist_valid else None,
            'logical_remote_dispatch_bytes':int(traffic.sum()-np.trace(traffic)) if hist_valid else None,
            'scope':'MEASURED_LOCAL_CUDA_SPANS_NOT_GLOBAL_WALL; LOGICAL_PAYLOAD_NOT_WIRE_OR_HBM_TRAFFIC'})
    return result


def component_summary(rows):
    grouped=defaultdict(list)
    for r in rows:grouped[(r['arm'],r['modality'])].append(r)
    result=[]
    for (arm,modality),items in grouped.items():
        out={'arm':arm,'modality':modality,'n':len(items)}
        for k in ('dispatch_total_ms','expert_total_ms','max_rank_expert_total_ms','combine_total_ms','moe_total_ms','communication_share',
                  'max_dispatch_quantum_ms','expert_reload_amp','expert_weight_bytes_proxy','logical_remote_dispatch_bytes'):
            vs=[r[k] for r in items if r[k] is not None]
            out[k]=None if not vs else {'mean':float(np.mean(vs)),'ci95':bootstrap(vs)}
        for k in ('rank_cumulative_moe_max_mean','max_iteration_accumulated_dispatch_ms'):
            vs=[r[k] for r in items if r[k] is not None]
            out[k]=None if not vs else {'mean':float(np.mean(vs)),'ci95':bootstrap(vs)}
        result.append(out)
    return result
