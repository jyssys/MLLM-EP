"""Same selection rule, different information; no expert migration/latency model."""
import argparse
import json
import numpy as np
import pandas as pd
from poc_agentic_ep.contract import ROOT,write
from poc_agentic_ep.metrics import choose_replicas,traffic,clustered


def memory_and_gap(replicas,expert_bytes,p2p):
    per_rank=replicas.sum(axis=(0,2))*expert_bytes
    edges=np.zeros((4,4),dtype=np.int64)
    for _,destination,expert in np.argwhere(replicas):edges[expert//32,destination]+=expert_bytes
    small={(r['source'],r['destination']):r['median_ms'] for r in p2p['rows'] if r['bytes']==expert_bytes}
    bulk={(r['source'],r['destination']):r['effective_gb_s'] for r in p2p['rows'] if r['bytes']==64*1024*1024}
    serial=sum(int(edges[s,d])//expert_bytes*small[s,d] for s in range(4) for d in range(4) if s!=d)
    bulk_serial=sum(int(edges[s,d])/bulk[s,d]/1e6 for s in range(4) for d in range(4) if s!=d)
    return dict(actual_prepared_experts=int(replicas.sum()),total_prepared_bytes=int(per_rank.sum()),
        max_replica_bytes_per_rank=int(per_rank.max()),prepared_bytes_per_rank=per_rank.tolist(),
        movement_bytes_by_edge=edges.tolist(),serial_copy_estimate_ms=float(serial),bulk_serial_estimate_ms=float(bulk_serial),
        min_tool_gap_estimate_ms=float(max(serial,bulk_serial)),
        gap_status={str(g):'FULLY_HIDDEN_ESTIMATE' if max(serial,bulk_serial)<=g else 'PARTIALLY_HIDDEN_ESTIMATE' for g in (50,100,250,500,1000)})


def main():
    p=argparse.ArgumentParser();p.add_argument('--version',required=True);a=p.parse_args();root=ROOT/'results'/a.version
    if (root/'preparation_summary.json').exists():raise FileExistsError('simulation immutable')
    frozen=json.loads((ROOT/'results/frozen_manifest_v2.json').read_text())
    pairs=json.loads((ROOT/'results/frozen_controls.json').read_text())['pairs']
    data=np.load(root/'aggregated.npz');ids=data['request_ids'].tolist();counts=data['source_expert_counts']
    index={rid:i for i,rid in enumerate(ids)};lookup={r['request_id']:r for r in frozen['rows']}
    calib=[i for i,rid in enumerate(ids) if lookup[rid]['split']=='calibration']
    global_counts=counts[calib].sum(axis=0)
    expert_bytes=json.loads((root/'loaded_expert_signature.json').read_text())['bytes_per_expert_per_layer']
    p2p=json.loads((root/'p2p.json').read_text())
    free=min(json.loads((root/f'runtime_memory.rank{r}.json').read_text())['free_bytes'] for r in range(4))
    memory_slots={str(g):int(g*1024**3//(48*expert_bytes)) for g in (.5,1,2,4)}
    budgets=sorted(set([1,2,4,*memory_slots.values()]))
    result=[];assignments=[]
    for pair in pairs:
        target=pair['target'];previous=pair['previous'];ti=index[target]
        with np.load(root/'raw_routing'/(target+'.npz')) as raw:
            all_ids=raw['expert_ids'];sources=raw['source_ep_rank'];category=raw['token_category']
            for pi,phase in enumerate(('new_prefill','decode')):
                mask=category==pi;routes=all_ids[:,mask,:];src=sources[mask]
                baseline=traffic(routes,src)
                for slots in budgets:
                    predictions={'global':global_counts[pi],'previous':counts[index[previous],pi],
                        'oracle':counts[ti,pi]}
                    for policy,predicted in predictions.items():
                        replicas=choose_replicas(predicted,slots);after=traffic(routes,src,replicas)
                        gap=memory_and_gap(replicas,expert_bytes,p2p)
                        row=dict(trajectory_id=pair['trajectory_id'],target=target,previous=previous,
                            phase=phase,policy=policy,slots=slots,nominal_max_bytes_per_rank=48*slots*expert_bytes,
                            feasible_memory=48*slots*expert_bytes<=free,**gap)
                        for key in ('remote_paths','remote_dispatch_bytes','remote_combine_bytes','max_rank_load','rank_max_mean','rank_cv','max_rank_remote_bytes'):
                            row['static_'+key]=baseline[key];row['after_'+key]=after[key]
                            row[key+'_reduction_pct']=100*(1-after[key]/baseline[key]) if baseline[key]>0 else None
                        result.append(row)
                        assignments.append(dict(target=target,previous=previous,phase=phase,policy=policy,slots=slots,
                            replica_triplets=np.argwhere(replicas).tolist(),scope='[layer,destination rank,expert ID]',
                            source_truth_used=policy=='oracle'))
    frame=pd.DataFrame(result);frame.to_csv(root/'preparation_budget.csv',index=False)
    summary={};contrasts={};recovery={}
    keys=['remote_paths_reduction_pct','remote_dispatch_bytes_reduction_pct','remote_combine_bytes_reduction_pct',
        'max_rank_load_reduction_pct','rank_max_mean_reduction_pct','rank_cv_reduction_pct','max_rank_remote_bytes_reduction_pct',
        'min_tool_gap_estimate_ms','max_replica_bytes_per_rank','total_prepared_bytes']
    for phase,part in frame.groupby('phase'):
        summary[phase]={};contrasts[phase]={};recovery[phase]={}
        for slots,g in part.groupby('slots'):
            summary[phase][str(slots)]={};contrasts[phase][str(slots)]={}
            for policy,h in g.groupby('policy'):
                summary[phase][str(slots)][policy]={k:clustered(h[k],h.trajectory_id) for k in keys}
            previous=g[g.policy=='previous'].set_index('target')
            oracle=g[g.policy=='oracle'].set_index('target').loc[previous.index]
            global_policy=g[g.policy=='global'].set_index('target').loc[previous.index]
            for comparator,h in [('global',global_policy),('oracle',oracle)]:
                contrasts[phase][str(slots)]['previous_minus_'+comparator]={k:clustered(previous[k]-h[k],previous.trajectory_id) for k in keys[:7]}
            denominator=oracle.remote_dispatch_bytes_reduction_pct
            ratio=100*previous.remote_dispatch_bytes_reduction_pct/denominator.replace(0,np.nan)
            good=ratio.notna()
            recovery[phase][str(slots)]=clustered(ratio[good],previous.loc[good,'trajectory_id']) if good.any() else dict(status='UNDEFINED_ZERO_ORACLE_BYTES')
    write(root/'preparation_summary.json',dict(metrics=summary,contrasts=contrasts,recovered_oracle_pct=recovery,
        memory_GiB_to_slots=memory_slots,measured_free_bytes_min=free,expert_bytes=expert_bytes,
        budget_definition='extra full expert replicas per layer per source EP rank; all48 layers count toward memory; idle sources do not prepare zero-score experts',
        oracle='assignment-optimal independent top slots per layer/source rank; byte headroom is a feasible next-truth policy, not a proven globally byte-optimal upper bound',
        static='exact actual owner32 with no replicas; unchanged token-expert paths',
        latency_model='UNVALIDATED_NOT_COMPUTED; no routed-byte-to-speedup conversion',
        gap_model='MEASURED_PEER_COPY_BASED_FEASIBILITY_ESTIMATE; no concurrent replication/packing/allocator/runtime mechanism measured',
        measured_latency_gain='NOT_RUN_NO_RUNTIME_METHOD'))
    write(root/'oracle_policy_assignments.json',dict(rows=assignments,
        leakage='global uses calibration only; previous uses prior request only; actual next-turn truth used solely by explicit nondeployable oracle'))
    print('PREPARATION_SIMULATION_COMPLETE')


if __name__=='__main__':main()
