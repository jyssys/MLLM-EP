"""Frozen causal/static/oracle plans; requests are independent statistics units."""
import collections
import concurrent.futures
import json
import argparse
import numpy as np
import pandas as pd
from poc_dllm_block_local_replica.contract import *
from poc_dllm_block_local_replica.metrics import closure_groups,post_traffic,load_stats
from poc_dllm_block_local_replica.scripts.destination_closure_oracle import bundle_greedy,lp_upper

POLICIES=('global','first_frequency','first_destination','first2_destination','oracle','oracle_load')


def load_blocks(info):
    result=[]
    with np.load(ROOT/'data/live_routes'/(info['request']+'.npz')) as z:
        for meta in info['blocks']:
            block=meta['block'];n=meta['iterations'];data={}
            for t in range(n):
                for l in range(1,20):
                    for s in range(4):
                        key=f'b{block}_t{t}_l{l}_s{s}_ids'
                        data[t,l,s]=z[key] if key in z else np.empty((0,8),dtype=np.int16)
            result.append((meta,data))
    return result


def groups_for(data,n,start,end):
    result={s:{l:collections.Counter() for l in range(1,20)} for s in range(4)}
    for t in range(start,min(end,n)):
        for l in range(1,20):
            for s in range(4):result[s][l].update(closure_groups(data[t,l,s],s))
    return result


def frequency(data,n,start,end):
    values=np.zeros((19,4,256),dtype=float)
    for t in range(start,min(end,n)):
        for l in range(1,20):
            for s in range(4):values[l-1,s]+=np.bincount(data[t,l,s].flatten(),minlength=256)
    return values


def choose_frequency(values,slots):
    result={s:{} for s in range(4)}
    for s in range(4):
        candidates=[(values[l-1,s,e],l,e) for l in range(1,20) for e in range(256)
                    if e//64!=s and values[l-1,s,e]>0]
        candidates.sort(key=lambda x:(-x[0],x[1],x[2]))
        for _,l,e in candidates[:slots]:result[s].setdefault(l,set()).add(e)
    return result


def make_plan(data,n,policy,slots,global_frequency):
    profile={'global':0,'first_frequency':1,'first_destination':1,'first2_destination':2,'oracle':0,'oracle_load':0}[policy]
    if profile>=n:return {s:{} for s in range(4)},profile,None
    if policy=='global':return choose_frequency(global_frequency,slots),profile,None
    if policy=='first_frequency':return choose_frequency(frequency(data,n,0,1),slots),profile,None
    observe=n if policy.startswith('oracle') else profile
    groups=groups_for(data,n,0,observe)
    plans={};bounds=[]
    current=np.zeros((n,19,4),dtype=np.int64)
    for t in range(n):
        for l in range(1,20):
            for s in range(4):current[t,l-1]+=np.bincount((data[t,l,s]//64).flatten(),minlength=4)
    cap=current.max(axis=2)*1.10
    for s in range(4):
        def allowed(l,experts):
            delta=np.zeros((n,4),dtype=np.int64)
            for t in range(profile,n):
                ids=data[t,l,s]
                for e in experts:
                    count=int((ids==e).sum());delta[t,s]+=count;delta[t,e//64]-=count
            changed=current[:,l-1]+delta
            if np.any(changed.max(axis=1)>cap[:,l-1]+1e-9):return False
            current[:,l-1]=changed
            return True
        plans[s]=bundle_greedy(groups[s],slots,allowed if policy=='oracle_load' else None)
        # LP only on primary smallest/largest budget; bound is optional, no
        # substitution of fractional plans for actually selected replicas.
        if policy=='oracle' and slots in (85,682):bounds.append(lp_upper(groups[s],slots))
    bound=sum(bounds) if bounds and all(x is not None for x in bounds) else None
    return plans,profile,bound


def evaluate_block(data,n,plans,profile):
    baseline=np.zeros((n,19,4),dtype=np.int64);after=baseline.copy()
    live=post=0;edges=np.zeros((4,4),dtype=np.int64);total_localized=0
    for t in range(n):
        for l in range(1,20):
            for s in range(4):
                ids=data[t,l,s];base=post_traffic(ids,s,set())
                chosen=plans[s].get(l,set()) if t>=profile else set()
                modified=post_traffic(ids,s,chosen)
                live+=base['dc_bytes'];post+=modified['dc_bytes']
                baseline[t,l-1]+=base['load'];after[t,l-1]+=modified['load']
                total_localized+=modified['localized_paths']
    replica_rows=[]
    for s in range(4):
        for l,experts in plans[s].items():
            for e in sorted(experts):
                edges[e//64,s]+=EXPERT_BYTES
                replica_rows.append(dict(layer=l,source_rank=s,remote_destination=e//64,expert=e,bytes=EXPERT_BYTES))
    premax=baseline.max(axis=2);postmax=after.max(axis=2)
    nonzero=premax>0
    before=baseline.sum(axis=(0,1));modified=after.sum(axis=(0,1))
    return dict(live_bytes=live,post_bytes=post,saved_bytes=live-post,copy_bytes=int(edges.sum()),
        per_edge_copy_bytes=edges.tolist(),extra_memory_per_rank=edges.sum(axis=0).tolist(),
        remaining_iterations=max(n-profile,0),profile_iterations=min(profile,n),iterations=n,
        baseline_load=before.tolist(),post_load=modified.tolist(),
        baseline_max_stage_sum=int(premax.sum()),post_max_stage_sum=int(postmax.sum()),
        worst_stage_load_increase=float((postmax[nonzero]/premax[nonzero]-1).max()) if nonzero.any() else 0.,
        localized_paths=total_localized,load_stats_baseline=load_stats(before),load_stats_post=load_stats(modified)),replica_rows


def process_request(args):
    info,global_frequency=args;results=[];allocations=[];blocks=[]
    loaded=load_blocks(info)
    for budget in BUDGETS:
        slots=int(budget*1024**3//EXPERT_BYTES)
        for policy in POLICIES:
            total=collections.Counter();baseline=np.zeros(4,dtype=np.int64);post=np.zeros(4,dtype=np.int64)
            peak=np.zeros(4,dtype=np.int64);worst=0.;upper=0.;hasbound=True
            for meta,data in loaded:
                plans,profile,bound=make_plan(data,meta['iterations'],policy,slots,global_frequency)
                stats,rows=evaluate_block(data,meta['iterations'],plans,profile)
                stats.update(request=info['request'],request_id=info['request_id'],domain=info['domain'],
                             block_length=info['block_length'],block=meta['block'],policy=policy,budget_gib=budget,
                             achieved_oracle_upper_bound_bytes=bound)
                blocks.append(stats)
                for row in rows:allocations.append(dict(request=info['request'],block=meta['block'],policy=policy,budget_gib=budget,**row))
                for key in ('live_bytes','post_bytes','saved_bytes','copy_bytes','baseline_max_stage_sum','post_max_stage_sum','localized_paths'):
                    total[key]+=stats[key]
                baseline+=np.array(stats['baseline_load']);post+=np.array(stats['post_load'])
                peak=np.maximum(peak,stats['extra_memory_per_rank']);worst=max(worst,stats['worst_stage_load_increase'])
                if bound is None:hasbound=False
                else:upper+=bound
            assert total['live_bytes']==info['B_live_only']
            assert peak.max()<=int(budget*1024**3)
            results.append(dict(request=info['request'],request_id=info['request_id'],domain=info['domain'],block_length=info['block_length'],
                policy=policy,budget_gib=budget,**total,B_one_shot=info['B_one_shot'],baseline_amp=info['B_live_only']/info['B_one_shot'],
                residual_amp=total['post_bytes']/info['B_one_shot'],dc_reduction=total['saved_bytes']/total['live_bytes'],
                byte_roi=total['saved_bytes']/total['copy_bytes'] if total['copy_bytes'] else 0.,
                peak_memory_gib=float(peak.max()/1024**3),max_rank_load_increase=float(post.max()/baseline.max()-1),
                stage_critical_load_increase=total['post_max_stage_sum']/total['baseline_max_stage_sum']-1,
                worst_stage_load_increase=worst,rank_max_mean=load_stats(post)['max_mean'],rank_cv=load_stats(post)['cv'],
                hottest_rank=load_stats(post)['hottest'],baseline_rank_assignments=baseline.tolist(),post_rank_assignments=post.tolist(),
                oracle_lp_upper_reduction=upper/total['live_bytes'] if hasbound else None))
    return results,blocks,allocations


def main():
    p=argparse.ArgumentParser();p.add_argument('--workers',type=int,default=4);a=p.parse_args()
    output=ROOT/'results/simulation'
    if output.exists():raise FileExistsError(output)
    output.mkdir()
    frozen=json.loads((ROOT/'results/design_freeze.json').read_text())
    manifest=json.loads((ROOT/'results/reused_trace_manifest.json').read_text())['requests']
    global_frequency=np.zeros((19,4,256))
    for info in manifest:
        if info['block_length']!=32 or info['request_id'] not in frozen['calibration']:continue
        counts=np.zeros_like(global_frequency)
        for meta,data in load_blocks(info):counts+=frequency(data,meta['iterations'],0,meta['iterations'])
        sums=counts.sum(axis=2,keepdims=True)
        global_frequency+=np.divide(counts,sums,out=np.zeros_like(counts),where=sums>0)/len(frozen['calibration'])
    np.save(output/'global_calibration_frequency.npy',global_frequency)
    results=[];blocks=[];allocations=[]
    with concurrent.futures.ProcessPoolExecutor(max_workers=a.workers) as pool:
        tasks=[pool.submit(process_request,(info,global_frequency)) for info in manifest]
        for task in concurrent.futures.as_completed(tasks):
            r,b,al=task.result();results.extend(r);blocks.extend(b);allocations.extend(al)
            print('SIMULATED',r[0]['request'],flush=True)
    df=pd.DataFrame(results).sort_values(['request','budget_gib','policy'])
    oracle=df[df.policy=='oracle'].set_index(['request','budget_gib'])
    df['oracle_recovery']=[r.saved_bytes/oracle.loc[(r.request,r.budget_gib)].saved_bytes
                           if oracle.loc[(r.request,r.budget_gib)].saved_bytes else 0. for r in df.itertuples()]
    df['split']=df.request_id.map(lambda r:'calibration' if r in frozen['calibration'] else 'heldout')
    df.to_csv(output/'request_policy.csv',index=False)
    pd.DataFrame(blocks).sort_values(['request','block','budget_gib','policy']).to_json(output/'block_policy.jsonl',orient='records',lines=True)
    pd.DataFrame(allocations).sort_values(['request','block','budget_gib','policy','source_rank','layer','expert']).to_csv(output/'layer_replica_plan.csv.gz',index=False,compression='gzip')
    new_json(output/'no_future_leakage.json',dict(global_calibration_ids=frozen['calibration'],
        heldout_ids=frozen['heldout'],disjoint=not(set(frozen['calibration'])&set(frozen['heldout'])),
        global_uses_block32_calibration_only=True,first_iteration_uses_current_block_ordinal0_only=True,
        first2_uses_ordinals0_1_only=True,profiling_traffic_unchanged=True,
        future_truth_policies=['oracle','oracle_load'],source_position_mapping_preserved=True,
        budget_shared_across_layers=True,dead_token_savings_in_replication=False))
    print('SIMULATION_COMPLETE',len(results),len(blocks),len(allocations),flush=True)


if __name__=='__main__':main()
