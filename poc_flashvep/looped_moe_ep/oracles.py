"""Measured-cost counterfactuals. No predicted value is a live E2E speedup."""
from collections import defaultdict
import json
import time
import numpy as np
from .common import *
from .analyze import rows,table,completed

def waterfill(counts,residents,world):
    """Integer exact-demand assignment to existing copies. Greedy, not optimal."""
    loads=np.zeros(world,dtype=int)
    flexible=[]
    for e,count in enumerate(counts):
        allowed=residents[e]
        if len(allowed)==1:loads[allowed[0]]+=int(count)
        else:flexible.append((int(count),allowed))
    for count,allowed in sorted(flexible,reverse=True):
        allowed=sorted(allowed,key=lambda r:(loads[r],r))
        left=count
        for width in range(1,len(allowed)+1):
            if width==len(allowed):
                q,rem=divmod(left,width)
                for r in allowed[:width]:loads[r]+=q
                for r in allowed[:rem]:loads[r]+=1
                break
            cost=(loads[allowed[width]]-loads[allowed[width-1]])*width
            if cost>left:
                q,rem=divmod(left,width)
                for r in allowed[:width]:loads[r]+=q
                for r in allowed[:rem]:loads[r]+=1
                break
            for r in allowed[:width]:loads[r]+=cost//width
            left-=cost
    assert loads.sum()==sum(counts)
    return loads

def objective(observations,residents,world):
    return np.mean([waterfill(c,residents,world).max() for c in observations])

def placement_proxy(observations,residents,world):
    # All calibration observations enter the same bounded static search.
    # Equal-copy splitting is a fitting proxy, not the exact online assignment.
    assignment=np.zeros((len(residents),world))
    for expert,ranks in enumerate(residents):
        assignment[expert,ranks]=1/len(ranks)
    return (np.asarray(observations)@assignment).max(1).mean()

def place(observations,world,budget,mode):
    experts=len(observations[0])
    resident=[[e%world] for e in range(experts)]
    history=[]
    for _ in range(budget):
        candidates=[]
        if mode=='global_hot':
            hot=np.sum(observations,axis=0)
            e=max(range(experts),key=lambda j:hot[j] if len(resident[j])==1 else -1)
            base=np.zeros(world)
            for j in range(experts):base[j%world]+=hot[j]
            rank=min([r for r in range(world) if r not in resident[e]],key=lambda r:base[r])
            candidates=[(e,rank)]
        else:
            candidates=[(e,r) for e in range(experts) for r in range(world) if r not in resident[e]]
        best=None
        for e,rank in candidates:
            trial=[v.copy() for v in resident]
            trial[e].append(rank)
            value=placement_proxy(observations,trial,world)
            if best is None or value<best[0]:best=(value,e,rank)
        _,e,rank=best
        resident[e].append(rank)
        history.append((e,rank))
    return resident,history

def replication_oracles():
    result=[];policies=[]
    for model in MODELS:
        for world in [4,8]:
            calibration=ROOT/'raw'/f'{model}_ep{world}_baseline_s0_v1'
            if not (calibration/'complete.json').exists():continue
            source=ROOT/'raw'/f'{model}_ep{world}_shape_quality_v2'/'routing_counts.jsonl'
            if not source.exists():continue
            calibration_rows=[r for r in rows(source) if r['split']=='calibration' and r['recurrent']]
            train=defaultdict(list)
            train_by_loop=defaultdict(lambda:defaultdict(list))
            for r in calibration_rows:
                train[r['site']].append(r['expert_counts'])
                train_by_loop[r['site']][r['loop']].append(r['expert_counts'])
            cache={}
            for site,observations in train.items():
                loop_means=[np.mean(v,axis=0).round().astype(int).tolist() for v in train_by_loop[site].values()]
                for budget in [1,world]:
                    generic,g_hist=place(observations,world,budget,'cost_aware')
                    hot,h_hist=place(observations,world,budget,'global_hot')
                    conditioned,l_hist=place(loop_means,world,budget,'cost_aware')
                    best_generic=min([generic,hot],key=lambda policy:objective(observations,policy,world))
                    cache[(site,budget)]=(generic,hot,conditioned,best_generic)
                    policies.append(dict(model=model,ep=world,budget=budget,site=site,
                        generic=g_hist,global_hot=h_hist,loop_conditioned=l_hist,
                        calibration='First 8 held-out IDs/domain reassigned to a disjoint placement-calibration split. No performance-cohort leakage.',
                        search='All calibration observations. Same equal-copy fitting proxy. Runtime uses exact-current-count waterfill. Not a global optimum.'))
            for batch in [1,2,4]:
                for length in [128,1024,2048]:
                    path=calibration/f'trace_b{batch}_p{length}.rank0.json'
                    calls=json.loads(path.read_text())
                    clean=np.mean(json.loads((calibration/f'timing_b{batch}_p{length}.rank0.json').read_text())['clean_forward_ms'])
                    by_site=defaultdict(list)
                    for r in calls:
                        if r['recurrent']:by_site[r['site']].append(r)
                    ranks=[json.loads((calibration/f'trace_b{batch}_p{length}.rank{r}.json').read_text()) for r in range(world)]
                    critical=max(range(world),key=lambda r:sum(v['total_span_ms'] for v in ranks[r]))
                    service={ (r['site'],r['loop']):r['expert_ms'] for r in ranks[critical]}
                    info=json.loads((calibration/'ownership.rank0.json').read_text())
                    for budget in [1,world]:
                        savings=defaultdict(float)
                        cpu_us=[];repl_bytes=0
                        for site,observed in by_site.items():
                            per_expert_bytes=next(v['per_expert_bytes'] for v in info if v['site']==site)
                            repl_bytes+=budget*per_expert_bytes
                            generic,hot,conditioned,best_generic=cache[(site,budget)]
                            for r in observed:
                                c=r['expert_counts'];base=max(r['rank_loads'])
                                t=service[(site,r['loop'])]
                                for name,resident in [('generic_hot',hot),('generic_cost',generic),('generic_best',best_generic),('loop_conditioned',conditioned)]:
                                    start=time.perf_counter()
                                    load=waterfill(c,resident,world)
                                    cpu_us.append((time.perf_counter()-start)*1e6)
                                    savings[name]+=t*(1-load.max()/max(base,1))
                                savings['omniscient']+=t*(1-np.mean(r['rank_loads'])/max(base,1))
                        for name in ['generic_hot','generic_cost','generic_best','loop_conditioned','omniscient']:
                            saved=savings[name]
                            speed=clean/max(clean-saved,clean*.01)
                            result.append(dict(model=model,ep=world,batch=batch,prefix=length,budget=budget,
                                policy=name,baseline_ms=clean,modeled_saved_expert_ms=saved,
                                modeled_speedup=speed,replica_total_bytes=repl_bytes if name!='omniscient' else None,
                                conservative_speedup=clean/max(clean-saved*.5+np.mean(cpu_us)/1000*len(calls),clean*.01),
                                selector_cpu_us=np.mean(cpu_us),online_model_includes_selector=False,
                                evidence='O1_MEMORY_CONSTRAINED_ROW_PROPORTIONAL_ESTIMATE' if name!='omniscient' else 'O0_UNRESIDENT_OMNISCIENT_LOAD_BOUND',
                                caveat='Held-out performance cohorts. Linear row-time mapping is optimistic. Communication held unchanged. Conservative case halves compute saving and charges selector CPU time. Not E2E measurement.'))
    table('oracle/replication.csv',result)
    write(ROOT/'oracle/replica_policies.json',policies)
    return result

def sample_cost(trace,timing):
    scale=np.mean(timing['clean_forward_ms'])/timing['observer_forward_ms']
    costs=[];previous=0.
    for r in trace:
        costs.append(dict(site=r['site'],loop=r['loop'],recurrent=r['recurrent'],
            dense=max(0,r['ready_forward_ms']-previous)*scale,expert=r['total_span_ms']*scale))
        previous=r['complete_forward_ms']
    tail=max(0,timing['observer_forward_ms']-previous)*scale
    return costs,tail

def simulate(costs,tails,batch,arrivals,window_ms,cross,decision_costs=None):
    """Single shared compute resource and ordered EP communicator. No future inputs."""
    pointers=[0]*batch
    awaiting={}
    done={}
    time_ms=0.;merges=0;strict=0;waits=[]
    def cost(kind,n,j):
        if n in costs:return costs[n][j][kind]
        return .5*(costs[2][j][kind]+costs[4][j][kind])
    while len(done)<batch:
        ready=[r for r in range(batch) if r not in done and r not in awaiting and arrivals[r]<=time_ms]
        if ready:
            first=min(ready,key=lambda r:(pointers[r],r))
            j=pointers[first]
            group=[r for r in ready if pointers[r]==j]
            duration=cost('dense',len(group),j)
            predicted=decision_costs[len(group)][j]['dense'] if decision_costs is not None else duration
            deadline=min((v['ready']+window_ms for v in awaiting.values()),default=float('inf'))
            if not awaiting or time_ms+predicted<=deadline:
                time_ms+=duration
                for r in group:awaiting[r]=dict(j=j,ready=time_ms)
                continue
        if awaiting:
            first=min(awaiting,key=lambda r:awaiting[r]['ready'])
            deadline=awaiting[first]['ready']+window_ms
            future=[arrivals[r] for r in range(batch) if r not in done and r not in awaiting and arrivals[r]>time_ms]
            if batch>1 and time_ms<deadline:
                if future and min(future)<=deadline:
                    time_ms=min(future)
                    continue
                # Fixed timeout pays for the wait even when no partner arrives.
                time_ms=deadline
            j=awaiting[first]['j']
            node=costs[1][j]
            group=[first]
            for r in awaiting:
                other=costs[1][awaiting[r]['j']]
                if r!=first and other['site']==node['site'] and ((cross and node['recurrent']) or other['loop']==node['loop']):
                    group.append(r)
            duration=max(cost('expert',len(group),awaiting[r]['j']) for r in group)
            if len(group)>1:
                merges+=len(group)-1
                strict+=sum(costs[1][awaiting[r]['j']]['loop']!=node['loop'] for r in group[1:])
            for r in group:waits.append(time_ms-awaiting[r]['ready'])
            time_ms+=duration
            for r in group:
                pointers[r]+=1
                del awaiting[r]
                if pointers[r]==len(costs[1]):done[r]=time_ms+tails.get(len(group),tails[1])
        else:
            time_ms=min(arrivals[r] for r in range(batch) if r not in done)
    return dict(makespan=max(done.values()),merges=merges,strict_cross_pass_merges=strict,
        mean_expert_queue_wait_ms=np.mean(waits),max_expert_queue_wait_ms=max(waits),
        ready_rows='Model-generated DAG readiness. Not observed asynchronous serving readiness.')

def coalescing_oracles():
    result=[];validation=[]
    for model in MODELS:
        for world in [4,8]:
            run=ROOT/'raw'/f'{model}_ep{world}_baseline_s0_v1'
            if not (run/'complete.json').exists():continue
            for length in [128,1024,2048]:
                costs={};tails={};observed={}
                for b in [1,2,4]:
                    t=json.loads((run/f'timing_b{b}_p{length}.rank0.json').read_text())
                    trace=json.loads((run/f'trace_b{b}_p{length}.rank0.json').read_text())
                    costs[b],tails[b]=sample_cost(trace,t)
                    observed[b]=np.mean(t['clean_forward_ms'])
                for b in [1,2,4]:
                    for pattern in ['burst','staggered','poisson']:
                        rng=np.random.default_rng(SEED)
                        gap=observed[1]/max(2*b,1)
                        arrivals=[0.]*b if pattern=='burst' else list(np.arange(b)*gap) if pattern=='staggered' else list(np.r_[0,np.cumsum(rng.exponential(gap,b-1))])
                        for window in [0.,.05,.1,.25,.5,1.,2.,5.,10.]:
                            generic=simulate(costs,tails,b,arrivals,window,False)
                            cross=simulate(costs,tails,b,arrivals,window,True)
                            result.append(dict(model=model,ep=world,batch=b,prefix=length,arrival=pattern,window_us=window*1000,
                                generic_ms=generic['makespan'],cross_ms=cross['makespan'],
                                cross_vs_generic_ratio=generic['makespan']/cross['makespan'],
                                strict_cross_pass_merges=cross['strict_cross_pass_merges'],
                                mean_queue_wait_ms=cross['mean_expert_queue_wait_ms'],
                                evidence='O1_DAG_CONSTRAINED_CALIBRATED_PREFILL_SIMULATION_NOT_LIVE',
                                scope='Complete native forward, not full generated request E2E. No asynchronous runtime validation.'))
                    modeled=simulate(costs,tails,b,[0.]*b,0,False)['makespan']
                    validation.append(dict(model=model,ep=world,batch=b,prefix=length,fit_error=modeled/observed[b]-1,
                        evidence='IN_SAMPLE_BURST_CALIBRATION_NOT_INDEPENDENT_VALIDATION'))
                for start in [1,2]:
                    heldout=ROOT/'raw'/f'{model}_ep{world}_baseline_s{start}_v1'/f'timing_b1_p{length}.rank0.json'
                    if heldout.exists():
                        t=json.loads(heldout.read_text())
                        actual=np.mean(t['clean_forward_ms'])
                        validation.append(dict(model=model,ep=world,batch=1,prefix=length,start=start,
                            fit_error=observed[1]/actual-1,evidence='HELD_OUT_PROCESS_START_WORKLOAD_SHAPE_CHECK'))
    table('oracle/coalescing.csv',result)
    table('oracle/replay_validation.csv',validation)
    return result

def main():
    a=coalescing_oracles()
    b=replication_oracles()
    c=generation_oracles()
    print('ORACLES',len(a),len(b),len(c))

def generation_oracles():
    result=[];validation=[]
    for model in MODELS:
        for world in [4,8]:
            base=ROOT/'raw'/f'{model}_ep{world}_baseline_s0_v1'
            if not (base/'complete.json').exists():continue
            cal={}
            for n in [1,2,4]:
                trace=json.loads((base/f'trace_b{n}_p128.rank0.json').read_text())
                timing=json.loads((base/f'timing_b{n}_p128.rank0.json').read_text())
                cal[n],_=sample_cost(trace,timing)
            cal[3]=[dict(row,dense=.5*(cal[2][j]['dense']+cal[4][j]['dense']),
                expert=.5*(cal[2][j]['expert']+cal[4][j]['expert'])) for j,row in enumerate(cal[1])]
            for start in [1,2]:
                run=ROOT/'raw'/f'{model}_ep{world}_baseline_s{start}_v1'
                path=run/'generation_trace.rank0.jsonl'
                if not (run/'complete.json').exists() or not path.exists():continue
                trace=rows(path)
                control=json.loads((run/'generation_trace_control.rank0.json').read_text())
                scale=control['clean']['wall_ms']/control['observer']['wall_ms']
                actual=[];previous=0.
                for row in trace:
                    actual.append(dict(site=row['site'],loop=row['loop'],recurrent=row['recurrent'],
                        dense=max(0,row['ready_generation_ms']-previous)*scale,
                        expert=row['total_span_ms']*scale))
                    previous=row['complete_generation_ms']
                tail=max(0,control['observer']['wall_ms']-previous)*scale
                costs={1:actual};pred={};tails={n:tail for n in [1,2,3,4]}
                for n in [1,2,3,4]:
                    pred[n]=[cal[n][j%len(cal[n])] for j in range(len(actual))]
                    if n==1:continue
                    costs[n]=[dict(row,dense=row['dense']*cal[n][j%len(cal[n])]['dense']/max(cal[1][j%len(cal[1])]['dense'],1e-6),
                        expert=row['expert']*cal[n][j%len(cal[n])]['expert']/max(cal[1][j%len(cal[1])]['expert'],1e-6))
                        for j,row in enumerate(actual)]
                for b in [1,2,4]:
                    for pattern in ['burst','staggered','poisson']:
                        rng=np.random.default_rng(SEED)
                        gap=control['clean']['wall_ms']/max(2*b,1)
                        arrival=[0.]*b if pattern=='burst' else list(np.arange(b)*gap) if pattern=='staggered' else list(np.r_[0,np.cumsum(rng.exponential(gap,b-1))])
                        for wait in [0.,.5,2.,5.]:
                            generic=simulate(costs,tails,b,arrival,wait,False,pred)
                            cross=simulate(costs,tails,b,arrival,wait,True,pred)
                            result.append(dict(model=model,ep=world,source_run=run.name,batch=b,arrival=pattern,window_us=wait*1000,
                                generic_ms=generic['makespan'],cross_ms=cross['makespan'],
                                modeled_ratio=generic['makespan']/cross['makespan'],strict_cross_pass_merges=cross['strict_cross_pass_merges'],
                                trace_steps=32,profile_normalization=scale,
                                evidence='O2_CAUSAL_POLICY_OFFLINE_PROFILE_REPLAY_NOT_LIVE_SPEEDUP',
                                limitation='Repeated representative request template. Group scaling from separate synchronized cohorts. Profile normalization is not independent baseline validation.'))
                validation.append(dict(model=model,ep=world,run=run.name,
                    raw_observer_ratio=control['observer_ratio'],
                    shape_lookup_32token_relative_error=32*sum(r['dense']+r['expert'] for r in cal[1])/control['clean']['wall_ms']-1,
                    scope='Prefill lookup versus first-occurrence generation. Warmed-generation follow-up is required.'))
    table('oracle/generation_coalescing.csv',result)
    table('oracle/generation_validation.csv',validation)
    return result

if __name__=='__main__':main()
