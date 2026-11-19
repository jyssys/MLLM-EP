"""Search legal same-cohort axes while preserving exact aggregate GEMM rows."""
import math
from .axis_oracle import KV_BLOCK_BUDGET,objective,demand_scores
from .replay import simulate,summary
from ..dense_ar_layer_token_pd.token_axis import matched_token_plan


def budget_suffixes(deltas):
    possible=[None]*(len(deltas)+1);possible[-1]={0}
    for j in range(len(deltas)-1,-1,-1):
        possible[j]=possible[j+1]|{v+deltas[j] for v in possible[j+1]}
    return possible


def close_budget(prefix,debt,deltas,possible,preferred):
    sequence=list(prefix)
    for j in range(len(prefix),len(deltas)):
        choices=(preferred,'Token' if preferred=='Layer' else 'Layer')
        for a in choices:
            next_debt=debt+(deltas[j] if a=='Token' else 0)
            if -next_debt in possible[j+1]:
                sequence.append(a);debt=next_debt;break
        else:raise AssertionError('No exact-work completion')
    assert debt==0
    return tuple(sequence)


def token_deltas(layers,count,batch):
    plan=matched_token_plan(layers,count,group_size=1)
    assert count%batch==0
    return [sum(36*s-8000*layers for s in plan[i:i+batch]) for i in range(0,count,batch)]


def evaluate(arrivals,sequence,table,batch,window,output,deltas,simulator=simulate):
    assert len(sequence)*batch==len(arrivals)
    assert sum(d for a,d in zip(sequence,deltas) if a=='Token')==0
    costs=[table[sequence[i//batch]] for i in range(len(arrivals))]
    rows=simulator(arrivals,costs,batch,window,
        kv_slots=KV_BLOCK_BUDGET//math.ceil((8000+output)/16))
    return summary(rows,output)


def beam(arrivals,table,layers,batch=1,window=32,output=16,width=32,deltas=None,seeds=(),simulator=simulate):
    """Recompute queue state for every feasible rollout, not frozen minima."""
    if deltas is None:deltas=token_deltas(layers,len(arrivals),batch)
    count=len(deltas);assert len(arrivals)==count*batch
    possible=budget_suffixes(deltas);assert sum(deltas)==0
    cache={}
    def measured(sequence):
        if sequence not in cache:
            cache[sequence]=evaluate(arrivals,sequence,table,batch,window,output,deltas,simulator)
        return cache[sequence]
    fixed={a:measured((a,)*count) for a in ('Layer','Token')}
    preferred=max(fixed,key=lambda a:objective(fixed[a]))
    candidates=[((),0)]
    for j in range(count):
        expanded=[]
        for prefix,debt in candidates:
            for a in ('Layer','Token'):
                next_debt=debt+(deltas[j] if a=='Token' else 0)
                if -next_debt not in possible[j+1]:continue
                next_prefix=prefix+(a,)
                rollout=close_budget(next_prefix,next_debt,deltas,possible,preferred)
                expanded.append((next_prefix,next_debt,objective(measured(rollout))))
        expanded.sort(key=lambda r:(r[2],r[0]),reverse=True)
        candidates=[(prefix,debt) for prefix,debt,_ in expanded[:width]]
        assert candidates
    final=[p for p,debt in candidates if debt==0]
    final.extend((a,)*count for a in fixed)
    for sequence in seeds:
        assert len(sequence)==count
        final.append(tuple(sequence))
    winner=max(final,key=lambda s:objective(measured(s)))
    return dict(fixed=fixed,best_fixed=preferred,oracle=measured(winner),oracle_sequence=winner,
        beam_width=width,scored_sequences=len(cache),cohort_size=batch,
        Token_row_deltas=deltas,final_row_delta=sum(d for a,d in zip(winner,deltas) if a=='Token'),
        evidence='OFFLINE_ORACLE',
        scope='Finite beam search, not a global optimal bound. Every chosen cohort has one axis. Aggregate P/D GEMM rows match exactly. Rollouts update FIFO queues, source window, full-cohort guard and uniform KV reservations. Actual future arrivals are available only to this oracle. Mixed-axis GPU execution and detailed hidden-buffer ownership are not qualified.')


def rule(arrivals,table,layers,kappa,batch=1,window=32,output=16,kind='two',deltas=None,simulator=simulate):
    if deltas is None:deltas=token_deltas(layers,len(arrivals),batch)
    possible=budget_suffixes(deltas);chosen={};decisions=[];debt=0
    reference='Layer';reference_cohort=batch*table[reference].preparation+table[reference].decode+table[reference].D_gap
    def choose(index,now,state):
        nonlocal debt
        group=index//batch
        if group not in chosen:
            pending=len(state['pending_source_indices']);ready=len(state['ready_indices'])
            active_fraction=state['estimated_active_D_remaining']/reference_cohort
            R=(1+ready/batch+active_fraction)/(1+pending)
            n=1+pending+ready+len(state['active_indices'])
            values=demand_scores(table,batch,n)
            if kind=='two':
                scores={a:r['score'] for a,r in values['candidates'].items()}
            else:
                assert kind=='one'
                scores={a:c.P+c.P_gap+kappa*R*(c.preparation+(c.decode+c.D_gap)/batch) for a,c in table.items()}
            feasible={a:-debt-(deltas[group] if a=='Token' else 0) in possible[group+1] for a in table}
            a=min((a for a in table if feasible[a]),key=lambda a:(scores[a],a))
            debt+=deltas[group] if a=='Token' else 0;chosen[group]=a
            decisions.append(dict(request_index=index,time=now,cohort=group,axis=a,scores=scores,
                pressure_ratio=R,total_unfinished=n,feasible_axes=feasible,row_debt=debt,
                Decode_demand_seconds=values['Decode_residency_per_request_seconds']))
        return table[chosen[group]]
    rows=simulator(arrivals,[table[reference]]*len(arrivals),batch,window,choose=choose,
        kv_slots=KV_BLOCK_BUDGET//math.ceil((8000+output)/16))
    assert debt==0
    return dict(metrics=summary(rows,output),sequence=[chosen[i] for i in range(len(deltas))],
        decisions=decisions,final_row_delta=debt,kind=kind,evidence='OFFLINE_REPLAY',
        scope='Causal-input shadow policy, not a GPU scheduler. The score reads only already-arrived host state and calibrated metadata. Exact-work feasibility uses the predeclared finite homogeneous request-index Token plan, not future arrival times or finishes. Row debt is fairness bookkeeping, not an extra performance knob. No production-stream generalization is implied.')
