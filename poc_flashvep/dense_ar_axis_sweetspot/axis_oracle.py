"""Compare fixed actions with a restricted state-updating axis oracle."""
import itertools
import math
from .replay import Cost,simulate,summary

KV_BLOCK_BUDGET=19113
KV_BUDGET_SOURCE='raw/state_Native_Last0_N8_B1_v2/KV_lifetime_decode.json'


def predicted(arrivals,axis_sequence,table,batch=1,window=32,output=16,block=18):
    assert len(arrivals)%block==0 and block%9==block%batch==0
    assert len(axis_sequence)==len(arrivals)//block
    costs=[table[axis_sequence[i//block]] for i in range(len(arrivals))]
    # The configured32-request source window cannot use more than33 KV slots.
    # Use the conservative full-prompt/output geometry, not partial-layer KV.
    blocks_per_request=math.ceil((8000+output)/16)
    kv_slots=KV_BLOCK_BUDGET//blocks_per_request
    rows=simulate(arrivals,costs,batch,window,kv_slots=kv_slots)
    return summary(rows,output)


def objective(metrics):
    return (metrics['qualified_output_tokens_s'],-metrics['TTFT_p95_ms'],metrics['output_tokens_s'])


def exhaustive(arrivals,table,batch=1,window=32,output=16,block=18):
    count=len(arrivals)//block
    assert len(arrivals)%block==0 and count<=8, 'Keep the restricted exhaustive search finite'
    fixed={a:predicted(arrivals,[a]*count,table,batch,window,output,block) for a in table}
    best_fixed=max(fixed,key=lambda a:objective(fixed[a]))
    best_sequence=None;best=None
    for sequence in itertools.product(sorted(table),repeat=count):
        value=predicted(arrivals,sequence,table,batch,window,output,block)
        if best is None or objective(value)>objective(best):
            best_sequence=sequence;best=value
    return dict(fixed=fixed,best_fixed=best_fixed,oracle=best,oracle_sequence=best_sequence,
        tested_sequences=len(table)**count,block=block,
        KV_budget_blocks=KV_BLOCK_BUDGET,KV_budget_source=KV_BUDGET_SOURCE,
        KV_model='Uniform full-prompt plus controlled-output reservations. This is not a per-layer transport or Hidden-buffer simulator.',
        evidence='OFFLINE_ORACLE',
        scope='Exhaustive only inside this fixed-cut,18-request-block action grid. Queues, cohort guards, source window and fixed-request KV reservations update after every action. This is not a global systems upper bound or measured online speedup. Mean Decode intervals do not establish measured ITL tail protection.')


def pressure_rule(arrivals,table,kappa=1.,batch=1,window=32,output=16,block=18):
    assert len(arrivals)%block==0 and block%9==block%batch==0
    reference=sorted(table)[0]
    reference_cohort=(batch*table[reference].preparation+table[reference].decode+table[reference].D_gap)
    chosen={};decisions=[]
    def choose(index,now,state):
        group=index//block
        if group not in chosen:
            P_pending=len(state['pending_source_indices'])
            D_pending=len(state['ready_indices'])/batch
            active_fraction=state['estimated_active_D_remaining']/reference_cohort
            pressure=(1+D_pending+active_fraction)/(1+P_pending)
            scores={a:(1+P_pending)*(c.P+c.P_gap)+kappa*(1+D_pending+active_fraction)*
                (c.preparation+(c.decode+c.D_gap)/batch) for a,c in table.items()}
            chosen[group]=min(scores,key=lambda a:(scores[a],a))
            decisions.append(dict(request_index=index,time=now,block=group,axis=chosen[group],
                P_pending=P_pending,D_pending_cohorts=D_pending,active_fraction=active_fraction,
                pressure_ratio=pressure,scores=scores,
                information='Only already-arrived pending P, already-ready D and calibrated active residual are read. Future events remain inside the simulator, outside the rule.'))
        return table[chosen[group]]
    rows=simulate(arrivals,[table[reference]]*len(arrivals),batch,window,choose=choose,
        kv_slots=KV_BLOCK_BUDGET//math.ceil((8000+output)/16))
    return dict(metrics=summary(rows,output),decisions=decisions,
        sequence=[chosen[i] for i in range(len(arrivals)//block)],kappa=kappa,
        evidence='OFFLINE_REPLAY',
        scope='A causal-input rule in a mean-cost simulator. This is not an implemented online scheduler. Both actions retain identical within-block GEMM work. Action costs are calibrated metadata, not additional online state variables. The reference D cost sets units and does not measure physical idle.')


def demand_scores(table,batch,unfinished):
    """Use common Decode demand and total work, not its queue location."""
    import statistics
    decode_per_request=statistics.median(c.decode for c in table.values())/batch
    alpha=max(0.,min(1.,(unfinished-batch)/(batch+1)))
    rows={}
    for a,c in table.items():
        p=c.P+c.P_gap
        suffix=c.preparation+c.D_gap/batch
        cycle=max(p,suffix+decode_per_request)
        unqueued=p+suffix
        rows[a]=dict(score=alpha*cycle+(1-alpha)*unqueued,
            cycle_seconds=cycle,unqueued_prefix_suffix_seconds=unqueued,
            P_effective_seconds=p,D_suffix_effective_seconds=suffix)
    return dict(Decode_residency_per_request_seconds=decode_per_request,
        total_unfinished_requests=unfinished,congestion_weight=alpha,candidates=rows)


def cycle_boundary(table,batch):
    """Describe the analytical crossing inside the calibrated max-stage model."""
    p={a:c.P+c.P_gap for a,c in table.items()}
    u={a:c.preparation+c.D_gap/batch for a,c in table.items()}
    if p['Token']<p['Layer'] and u['Layer']<u['Token']:
        return dict(case='TOKEN_FASTER_P_LAYER_FASTER_D',
            common_Decode_demand_threshold_seconds=p['Layer']-u['Token'],
            below='Token',above='Layer',P_effective_seconds=p,D_suffix_effective_seconds=u,
            evidence='STRUCTURAL_ACCOUNTING',
            scope='Exact crossing of max(p_axis,u_axis+d) for positive d in this simplified model. A nonpositive threshold has no positive-d Token region. This is not a proven online TTFT or SLO boundary.')
    if p['Layer']<p['Token'] and u['Token']<u['Layer']:
        return dict(case='LAYER_FASTER_P_TOKEN_FASTER_D',
            common_Decode_demand_threshold_seconds=p['Token']-u['Layer'],
            below='Layer',above='Token',P_effective_seconds=p,D_suffix_effective_seconds=u,
            evidence='STRUCTURAL_ACCOUNTING',scope='Only the calibrated max-stage model is described.')
    dominant=('Layer' if p['Layer']<=p['Token'] and u['Layer']<=u['Token'] else
        'Token' if p['Token']<=p['Layer'] and u['Token']<=u['Layer'] else 'TIE')
    return dict(case='NO_STRICT_TRADEOFF',dominant=dominant,P_effective_seconds=p,
        D_suffix_effective_seconds=u,evidence='STRUCTURAL_ACCOUNTING',
        scope='A dominance statement inside the calibrated mean-cost model, not a measured online result.')


def demand_rule(arrivals,table,batch=1,window=32,output=16,block=None):
    if block is None:
        block=math.lcm(18,batch)
    assert len(arrivals)%block==0
    chosen={};decisions=[]
    def choose(index,now,state):
        group=index//block
        if group not in chosen:
            unfinished=1+len(state['pending_source_indices'])+len(state['ready_indices'])+len(state['active_indices'])
            result=demand_scores(table,batch,unfinished)
            chosen[group]=min(result['candidates'],key=lambda a:(result['candidates'][a]['score'],a))
            decisions.append(dict(request_index=index,time=now,block=group,axis=chosen[group],**result,
                information='Total unfinished counts only already-arrived requests. Decode demand comes from calibration, not a future measured completion.'))
        return table[chosen[group]]
    reference=sorted(table)[0]
    rows=simulate(arrivals,[table[reference]]*len(arrivals),batch,window,choose=choose,
        kv_slots=KV_BLOCK_BUDGET//math.ceil((8000+output)/16))
    return dict(metrics=summary(rows,output),decisions=decisions,
        sequence=[chosen[i] for i in range(len(arrivals)//block)],block=block,
        evidence='OFFLINE_REPLAY',
        scope='Prespecified two-scalar surrogate: calibrated Decode occupancy per request plus total unfinished work. It is invariant to a pure relocation between P/D waiting queues. No online scheduler, deadline guarantee or physical GPU-idle estimate is implemented.')
