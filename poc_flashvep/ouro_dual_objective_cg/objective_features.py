"""Causal occupancy-risk proxies. They do not predict a future complete-token ITL."""
import math
from .common import PRIOR
from ..ouro_budget_joint_cg.cost_predictor import context_bin

def features(candidate,state,normalization,model):
    c=candidate
    n=len(state.get('issued_decode',[]))
    exposure=max(c.current_ms,c.opening_ms)
    efficiency=c.C/max(.001,c.residual_chunk_ms)
    target=state.get('TBT_target_ms',normalization['reference_TBT_ms'])
    age=state.get('decode_age_max_ms',0.)
    # No complete-token prediction. Age is observed, exposure is a planned
    # nonpreemptible-unit estimate, and remaining decode work stays unresolved.
    risk=0. if n==0 else exposure/max(1.,target)*(1+min(2.,age/max(1.,target)))
    path=c.path
    point,point_known=model.raw(path,c.C,c.G,state['start']+c.C,state.get('depth',0))
    width=max(0.,c.current_ms-point)
    tail_key=f'{path}:{c.G}'
    tail_ratio=normalization['tail_ratio'].get(tail_key,1.5)
    tail_risk=0. if n==0 else risk*tail_ratio
    yield_cost=max(0,c.groups_remaining-1)*normalization['submission_cost_ms']
    retained_MiB=0. if path=='atomic' else c.C*2048*2/2**20
    E=efficiency/max(.001,normalization['efficiency_scale'])
    R=risk/max(.001,normalization['risk_scale'])
    return dict(E=E,R=R,efficiency_tokens_ms=efficiency,planned_exposure_ms=exposure,
        estimated_full_R4_ms=c.residual_chunk_ms,complete_now=c.complete_tokens_now,eventual_tokens=c.C,
        tail_R=tail_risk/max(.001,normalization['risk_scale']),
        yield_fraction=yield_cost/max(.001,c.residual_chunk_ms),retained_MiB=retained_MiB,
        memory_n=retained_MiB/8.,uncertainty_n=width/max(1.,normalization['service_scale_ms'])+float(not c.known),
        pressure=min(2.,n/4),prompt_age_n=min(3.,state.get('prompt_age_ms',0.)/2000.),
        feedback_violation=state.get('past_TBT_violation_fraction',0.),
        feedback_samples=state.get('past_TBT_samples',0),
        switching_C=0. if not state.get('previous_C') else abs(math.log2(c.C/state['previous_C'])),
        switching_G=0. if not state.get('previous_G') else abs(c.G-state['previous_G'])/3,
        support='EMPIRICAL_SHAPE' if c.known else 'ENCLOSING_OR_INFLATED_SHAPE',
        risk_kind='CAUSAL_PLANNED_EXPOSURE_PROXY_NOT_P_ITL_NOT_DEADLINE_BOUND')

def prepare_items(candidates,state,normalization,model):
    """Construct features before restricting C-only/G-only action surfaces."""
    items=[(c,features(c,state,normalization,model)) for c in candidates]
    for c,f in items:
        f['axis_E_C']=sum(x['E'] for b,x in items if b.C==c.C)/sum(b.C==c.C for b,_ in items)
        f['axis_R_G']=sum(x['R'] for b,x in items if b.G==c.G)/sum(b.G==c.G for b,_ in items)
    return items

def contextualize_offline(state,cohort,clock):
    """Use only delivered outputs at or before the historical decision clock."""
    result=dict(state)
    eligible={r['id'] for r in state.get('issued_decode',[])}
    ages=[];past=[]
    for request in cohort['requests']:
        delivered=[t for t in request['token_times'] if t<=clock]
        if request['id'] in eligible and delivered:
            ages.append(1000*(clock-delivered[-1]))
        past.extend(1000*(b-a) for a,b in zip(delivered,delivered[1:]))
        if request['id']==state['id']:
            result['prompt_age_ms']=max(0.,1000*(clock-request['engine_enqueue']))
    result.update(decode_age_max_ms=max(ages,default=0.),
        past_TBT_samples=len(past),past_TBT_violation_fraction=sum(x>200. for x in past)/max(1,len(past)),
        TBT_target_ms=200.,telemetry_scope='Historical causal clock reconstruction. No future output values used.')
    return result
