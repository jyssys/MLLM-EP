"""Matched-memory/load, warm-global versus request-specific cold-once plans."""
import argparse
import concurrent.futures
import json
import numpy as np
import pandas as pd
from poc_dllm_request_persistent_replica.contract import *
from poc_dllm_request_persistent_replica.data import load_profile
from poc_dllm_request_persistent_replica.planner import plan,global_inputs

CAPS=(.10,.15,.20,None)
POLICIES=('global_frequency','global_destination','first_block','first2_blocks','first_frequency','first2_frequency','oracle','oracle_load')


def plan_to_json(plans):
    return {str(s):{str(l):sorted(experts) for l,experts in layers.items()} for s,layers in plans.items()}


def process(job):
    info,global_plans=job;p=load_profile(info);rows=[];block_rows=[];allocation=[]
    histories={1:p.subset(0,1),2:p.subset(0,2),0:p}
    for budget in BUDGETS:
        slots=int(budget*1024**3//EXPERT_BYTES)
        for cap in CAPS:
            for policy in POLICIES:
                if policy=='oracle' and cap is not None:continue
                if policy=='oracle_load' and cap is None:continue
                profile=1 if policy in ('first_block','first_frequency') else 2 if policy in ('first2_blocks','first2_frequency') else 0
                if policy.startswith('global'):
                    selected=global_plans[(policy,budget,cap)]
                elif profile>=p.nblocks:
                    selected={s:{} for s in range(4)}
                else:
                    selected=plan(histories[profile],slots,cap,
                                  frequency_only=policy in ('first_frequency','first2_frequency'))
                result=p.evaluate(selected,profile)
                if policy.startswith('global'):
                    result['resident_weight_bytes']=result['copy_bytes']
                    result.update(copy_bytes=0,initial_copy_bytes=0,update_bytes=0,byte_roi=None,
                                  copy_edges=np.zeros((4,4),dtype=np.int64).tolist(),byte_break_even_block=None,byte_break_even_iteration=None)
                else:result['resident_weight_bytes']=result['copy_bytes']
                assert result['peak_memory_gib']<=budget+1e-10
                key=dict(request=info['request'],request_id=info['request_id'],lane=info['lane'],domain=info['domain'],
                         split=info['split'],budget_gib=budget,load_guard=cap if cap is not None else -1.,policy=policy)
                compact={k:v for k,v in result.items() if k not in ('replica_use','block_baseline_bytes','block_post_bytes','per_iteration_saved_bytes')}
                compact.update(key,B_one_shot=info['B_one_shot'],baseline_amp=info['B_live_only']/info['B_one_shot'],
                               residual_amp=result['post_bytes']/info['B_one_shot'],dc_reduction=result['saved_bytes']/result['live_bytes'],
                               nblocks=p.nblocks,niterations=p.n,
                               plan=plan_to_json(selected))
                rows.append(compact)
                for block,(base,after) in enumerate(zip(result['block_baseline_bytes'],result['block_post_bytes'])):
                    block_rows.append(dict(**key,block=block,baseline_bytes=base,post_bytes=after,
                                           saved_bytes=base-after,profiled_block=block<profile,iterations=p.iterations[block]))
                for r in result['replica_use']:allocation.append(dict(**key,**r))
    return rows,block_rows,allocation


def main():
    q=argparse.ArgumentParser();q.add_argument('--lane',choices=['short','long'],required=True);q.add_argument('--workers',type=int,default=4);q.add_argument('--revision',default='v2');a=q.parse_args()
    out=ROOT/'results'/('simulation_'+a.lane+'_'+a.revision)
    if out.exists():raise FileExistsError(out)
    out.mkdir(parents=True)
    infos=json.loads((ROOT/'results'/(a.lane+'_manifest.json')).read_text())['requests']
    cal=[load_profile(i) for i in infos if i['split']=='calibration']
    history,groups,freq=global_inputs(cal)
    global_plans={}
    for budget in BUDGETS:
        slots=int(budget*1024**3//EXPERT_BYTES)
        for cap in CAPS:
            for policy in ('global_frequency','global_destination'):
                global_plans[policy,budget,cap]=plan(history,slots,cap,groups_override=groups,freq_override=freq,
                                                   frequency_only=policy=='global_frequency')
                print('GLOBAL_PLAN',a.lane,policy,budget,cap,flush=True)
    new_json(out/'global_plans.json',dict(plans={f'{p}@{b:g}@{c}':plan_to_json(s) for (p,b,c),s in global_plans.items()},
        calibration_ids=[i['request_id'] for i in infos if i['split']=='calibration'],
        heldout_ids=[i['request_id'] for i in infos if i['split']=='heldout'],prewarmed_copy_cost=0))
    np.save(out/'global_frequency.npy',freq)
    rows=[];blocks=[];allocation=[]
    with concurrent.futures.ProcessPoolExecutor(max_workers=a.workers) as pool:
        jobs=[pool.submit(process,(i,global_plans)) for i in infos]
        for job in concurrent.futures.as_completed(jobs):
            r,b,al=job.result();rows.extend(r);blocks.extend(b);allocation.extend(al)
            print('PERSIST_SIMULATED',a.lane,r[0]['request'],flush=True)
    df=pd.DataFrame(rows).sort_values(['request','budget_gib','load_guard','policy'])
    df.to_json(out/'request_policy.jsonl',orient='records',lines=True)
    df.drop(columns=['plan','copy_edges','baseline_rank_assignments','rank_assignments']).to_csv(out/'policy_budget.csv',index=False)
    pd.DataFrame(blocks).sort_values(['request','budget_gib','load_guard','policy','block']).to_csv(out/'block_policy.csv',index=False)
    pd.DataFrame(allocation).sort_values(['request','budget_gib','load_guard','policy','layer','source','expert']).to_csv(out/'replica_allocation.csv.gz',index=False,compression='gzip')
    new_json(out/'causal_audit.json',dict(causal_policies=['first_block','first2_blocks','first_frequency','first2_frequency'],
        global_uses_only_completed_calibration=True,current_request_decode_future_not_used=True,
        caps_enforced_on_observed_history_not_future=True,heldout_truth_not_used_for_pruning=True,
        profiling_full_blocks_charged=True,copy_once_per_request=True,incremental_updates=0,
        future_truth_policies=['oracle','oracle_load'],load_guard_curves=[10,15,20],unconstrained_reference=True,
        global_static_is_prewarmed=True,dead_token_savings_not_counted=True))
    print('PERSIST_SIMULATION_COMPLETE',a.lane,len(rows),len(blocks),len(allocation),flush=True)


if __name__=='__main__':main()
