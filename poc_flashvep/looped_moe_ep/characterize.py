"""Additional measured accounting and explicitly qualified structural controls."""
from collections import defaultdict
import json
import numpy as np
from .common import ROOT, MODELS
from .analyze import rows,table
from .finalize import md,save,LABEL

def main():
    records=[];overlap=[];memory=[];alias=[];routes=[];scaling=[];placement_memory=[];warmed_validation=[]
    for model in MODELS:
        for world in [4,8]:
            run=ROOT/'raw'/f'{model}_ep{world}_baseline_s0_v1'
            for b in [1,2,4]:
                for length in [128,1024,2048]:
                    allr=[json.loads((run/f'trace_b{b}_p{length}.rank{r}.json').read_text()) for r in range(world)]
                    critical=max(range(world),key=lambda r:sum(x['total_span_ms'] for x in allr[r]))
                    selected=allr[critical]
                    times=json.loads((run/f'timing_b{b}_p{length}.rank0.json').read_text())
                    for recurrent in [False,True]:
                        group=[x for x in selected if x['recurrent']==recurrent]
                        if not group:continue
                        cv=[];maxmean=[];hot=[]
                        for x in group:
                            load=np.array(x['rank_loads'],float)
                            cv.append(load.std()/load.mean())
                            maxmean.append(load.max()/load.mean())
                            hot.append(int(np.argmax(x['expert_counts'])))
                        span=sum(x['total_span_ms'] for x in group)
                        expert=sum(x['expert_ms'] for x in group)
                        fraction=expert/times['observer_forward_ms']
                        records.append(dict(model=model,ep=world,batch=b,prefix=length,recurrent=recurrent,
                            logical_calls=len(group),mean_rank_row_cv=np.mean(cv),mean_rank_row_max_mean=np.mean(maxmean),
                            expert_api_span_ms=expert,complete_ep_api_span_ms=span,
                            expert_api_fraction_of_observer=fraction,
                            ideal_erase_expert_api_ratio=1/max(1-fraction,.01),
                            evidence='LIVE_API_SPANS_AND_ROUTER_COUNTS_WITH_STRUCTURAL_ERASURE_NOT_CLEAN_CRITICAL_PATH'))
                    history={}
                    info=json.loads((run/'ownership.rank0.json').read_text())
                    for x in allr[0]:
                        if not x['recurrent']:continue
                        site=x['site']
                        previous=history.get(site)
                        if previous is not None:
                            a=previous['sampled_ids'];z=x['sampled_ids']
                            if len(a)==len(z):
                                false=sum(len(set(pr)-set(actual)) for pr,actual in zip(a,z))
                                predicted=set(e for ids in a for e in ids)
                                actual=set(e for ids in z for e in ids)
                                bytes_per=next(v['per_expert_bytes'] for v in info if v['site']==site)
                                routes.append(dict(model=model,ep=world,batch=b,prefix=length,site=site,loop=x['loop'],
                                    sampled_positions=len(a),false_positive_edges=false,predicted_edges=2*len(a),
                                    unused_predicted_unique_experts=len(predicted-actual),
                                    potential_unused_replica_bytes=len(predicted-actual)*bytes_per,
                                    evidence='LIVE_SAMPLED_ROUTES_STRUCTURAL_PREDICTED_COPY_COST_NO_PREFETCH'))
                        history[site]=x
            # Same resident set and same exact current routing, with no pass argument.
            from .oracles import waterfill
            policies=json.loads((ROOT/'oracle/replica_policies.json').read_text())
            checks=0
            for policy in policies:
                if policy['model']!=model or policy['ep']!=world:continue
                owned=json.loads((run/'ownership.rank0.json').read_text())
                e=next(v['logical_experts'] for v in owned if v['site']==policy['site'])
                resident=[[j%world] for j in range(e)]
                for expert,rank in policy['loop_conditioned']:resident[expert].append(rank)
                trace=json.loads((run/'trace_b4_p2048.rank0.json').read_text())
                for event in trace:
                    if event['site']!=policy['site']:continue
                    generic=waterfill(event['expert_counts'],resident,world)
                    conditioned=waterfill(event['expert_counts'],resident,world)
                    assert np.array_equal(generic,conditioned)
                    checks+=1
            alias.append(dict(model=model,ep=world,checks=checks,assignment_exact=True,
                ratio=1.0,evidence='CPU_SELECTOR_IDENTITY_CONTROL_NOT_GPU_SPEEDUP',
                qualification='Same selector, same residents, same current routes. This tests pass-label information only.'))
            for budget in [1,world]:
                selected=[p for p in policies if p['model']==model and p['ep']==world and p['budget']==budget]
                for name in ['generic','global_hot','loop_conditioned']:
                    extra=np.zeros(world,dtype=np.int64)
                    for policy in selected:
                        size=next(v['per_expert_bytes'] for v in owned if v['site']==policy['site'])
                        for _,rank in policy[name]:extra[rank]+=size
                    peaks=[max(json.loads(p.read_text())['peak_allocated_bytes'] for p in run.glob(f'timing_b*_p*.rank{rank}.json')) for rank in range(world)]
                    projection=np.array(peaks)+extra
                    placement_memory.append(dict(model=model,ep=world,budget=budget,policy=name,
                        extra_bytes_per_rank=extra.tolist(),total_extra_bytes=int(extra.sum()),
                        max_projected_allocated_bytes=int(projection.max()),
                        fits_recorded_physical_hbm=bool(projection.max()<81559*1024**2),
                        evidence='STRUCTURAL_CANDIDATE_COPY_BYTES_PLUS_MEASURED_PEAK_NOT_LIVE_COMBINED_PEAK'))
            for source in [run,ROOT/'raw'/f'{model}_ep{world}_shape_quality_v2']:
                if not (source/'complete.json').exists():continue
                operators=[rows(source/f'overlap.rank{r}.jsonl')[0] for r in range(world)]
                seq=max(np.mean(r['sequential_ms']) for r in operators)
                par=max(np.mean(r['overlap_ms']) for r in operators)
                exact=[r.get('expert_output_exact') for r in operators]
                overlap.append(dict(model=model,ep=world,run=source.name,sequential_max_rank_ms=seq,
                    two_stream_max_rank_ms=par,operator_ratio=seq/par,
                    outputs_exact=all(exact) if all(v is not None for v in exact) else 'NOT_RECORDED_OLDER_CONTROL',
                    request_e2e_saved_ms=None,evidence='LIVE_OPERATOR_COMPLETE_WALL_CONTROL_NOT_REQUEST_E2E'))
                sample=defaultdict(list)
                for rank in range(world):
                    for record in rows(source/f'operator.rank{rank}.jsonl'):
                        sample[(record['batch'],record['input_tokens'],record['factor'])].append(record)
                for (batch,length,factor),values in sample.items():
                    duration=max(np.mean(r['ms']) for r in values)
                    single=max(np.mean(r['ms']) for r in sample[(batch,length,1)])
                    scaling.append(dict(model=model,ep=world,run=source.name,batch=batch,prefix=length,
                        factor=factor,max_rank_operator_ms=duration,
                        local_gemm_batching_ratio=factor*single/duration,
                        evidence='LIVE_NATIVE_WEIGHT_GEMM_SHAPE_CONTROL_NOT_READY_COALESCING_OR_E2E'))
            allocation=[v for rank in range(world) for v in rows(run/f'replica_memory.rank{rank}.jsonl')]
            for budget in [1,world]:
                selected=[r for r in allocation if r['extra_copies_per_recurrent_site']==budget]
                memory.append(dict(model=model,ep=world,copies_per_recurrent_site=budget,
                    total_retained_bytes=sum(r['retained_tensor_bytes'] for r in selected),
                    max_rank_retained_bytes=max(r['retained_tensor_bytes'] for r in selected),
                    max_rank_allocated_delta_bytes=max(r['allocated_delta_bytes'] for r in selected),
                    max_initialization_seconds=max(r['initialization_seconds'] for r in selected),
                    exact_hashes=all(r['exact_replica_hashes'] for r in selected),
                    qualification='Controlled fixed copy layout, not the selected candidate placement.',evidence='LIVE_REPLICA_COPY_AND_MEMORY_CONTROL'))
            steady=ROOT/'raw'/f'{model}_ep{world}_steady_generation_v1'
            if (steady/'complete.json').exists():
                short=next(r for r in rows(steady/'steady_generation.rank0.jsonl') if r['cap']==32)
                forward=np.mean(json.loads((run/'timing_b1_p128.rank0.json').read_text())['clean_forward_ms'])
                warmed_validation.append(dict(model=model,ep=world,lookup_32token_ms=32*forward,
                    actual_warmed_32token_ms=1000*short['e2e_s'],relative_error=32*forward/(1000*short['e2e_s'])-1,
                    evidence='HELD_OUT_PROCESS_START_WARMED_LOOKUP_CHECK',
                    qualification='Matched prefix cohort, but growing generation context and varying routing. Not asynchronous schedule validation.'))
    table('summary/characterization.csv',records)
    table('summary/overlap_cohort.csv',overlap)
    table('summary/replica_copy_costs.csv',memory)
    table('summary/route_false_positive_cost.csv',routes)
    table('summary/operator_scaling.csv',scaling)
    table('oracle/placement_memory.csv',placement_memory)
    table('oracle/warmed_generation_validation.csv',warmed_validation)
    table('oracle/same_resident_set_control.csv',alias)
    save('characterization.md',f'''# EP characterization and memory

## Recurrent versus one-off execution

The following table uses B1 and prefix 128.
Rank imbalance comes from exact executed top-2 routing counts.
API spans come from one cumulative-critical rank in a separate observer.
They are not clean E2E critical-path percentages.

{md(['Model','EP','Core?','Calls','Mean rank CV','Mean max/mean','Expert span ms','Complete EP span ms'],[[LABEL[r['model']],r['ep'],r['recurrent'],r['logical_calls'],f'{r["mean_rank_row_cv"]:.3f}',f'{r["mean_rank_row_max_mean"]:.3f}',f'{r["expert_api_span_ms"]:.2f}',f'{r["complete_ep_api_span_ms"]:.2f}'] for r in records if r['batch']==1 and r['prefix']==128])}

Unequal rank row counts do not directly equal unequal GPU execution times.
Small expert batches retain launch and grouping costs.
The replica model's linear row-time conversion is therefore optimistic.
Huginn and Foil one-off calls remain in every whole-model cost.
The study cannot remove those calls through recurrent-core placement.

## Real replica memory and initialization

{md(['Model','EP','Extra copies/site','Total GiB','Max rank GiB','Init max-rank s'],[[LABEL[r['model']],r['ep'],r['copies_per_recurrent_site'],f'{r["total_retained_bytes"]/2**30:.3f}',f'{r["max_rank_retained_bytes"]/2**30:.3f}',f'{r["max_initialization_seconds"]:.3f}'] for r in memory])}

The control allocates original BF16 expert tensors and verifies matching hashes across ranks.
It does not execute replica-aware dispatch.
The copy layout is fixed for memory validation.
Candidate placement bytes are calculated separately from exact per-expert tensor sizes.
No candidate assigns work to an absent modeled expert copy.

## Compute/communication operator control

{md(['Model','EP','Start','Sequential ms','Two streams ms','Operator ratio','Output check'],[[LABEL[r['model']],r['ep'],r['run'].split('_ep')[1],f'{r["sequential_max_rank_ms"]:.4f}',f'{r["two_stream_max_rank_ms"]:.4f}',f'{r["operator_ratio"]:.3f}×',r['outputs_exact']] for r in overlap])}

The ratio uses the maximum rank-local mean in each complete operator condition.
Each condition has ten warmups and twenty measurements.
The two source runs use separate process starts.
The original control did not record an operator output comparison.
The later quality control adds that check and passes it.
The control tests independent native tensors, not two live asynchronous requests.
Request-level critical-path saving remains NOT_MEASURED.
Most tested controls regress when the streams compete for GPU resources.
Isolated positive rank results do not justify an E2E claim.

## Route lookahead

The prior-pass diagnostic compares authoritative top-2 routes at sampled positions.
It never changes routing or dispatches future activations.
`summary/route_false_positive_cost.csv` records false-positive edges and unused predicted expert sets.
Potential copy bytes describe a structural diagnostic, not actual allocated prefetch memory.
The gate needs the next pass's actual hidden state.
Observed route similarity cannot remove that dependency.
''')
    print('CHARACTERIZATION',len(records),len(overlap),len(memory),len(alias))

if __name__=='__main__':main()
