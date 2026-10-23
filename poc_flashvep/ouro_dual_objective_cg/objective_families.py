"""Fourteen transparent objective families on one identical legal surface."""
import math

NAMES={
 'O0':'Original full-R4 efficiency', 'O1':'Separable axis diagnostic',
 'O2':'Coupled normalized weighted sum','O3':'Risk epsilon constraint',
 'O4':'Risk-class lexicographic','O5':'Minimum efficiency then risk',
 'O6':'Observed TTFT/decode pressure weights','O7':'Empirical service-tail proxy',
 'O8':'Pareto ideal-distance selection','O9':'Efficiency/risk geometric utility',
 'O10':'Yield-aware full-R4 service','O11':'Uncertainty-penalized dual',
 'O12':'Switching-aware dual','O13':'Past-gap feedback dual'}

def scores(family,items,weight=1.,endpoint='DUAL',risk_cap=.5,efficiency_floor=.5):
    """Items pair candidates and frozen features. Return ordered score tuples."""
    assert family in NAMES
    if endpoint=='E_ONLY': return [(f['E'],) for _,f in items]
    if endpoint=='R_ONLY': return [(-f['R'],f['E']) for _,f in items]
    assert endpoint=='DUAL'
    pareto=[]
    if family=='O8':
        for i,(_,f) in enumerate(items):
            dominated=any(g['E']>=f['E'] and g['R']<=f['R'] and (g['E']>f['E'] or g['R']<f['R']) for _,g in items)
            if not dominated: pareto.append(i)
    eligible_risk=any(f['R']<=risk_cap for _,f in items)
    eligible_eff=any(f['E']>=efficiency_floor for _,f in items)
    minimum_class=min((math.floor(f['R']*4) for _,f in items),default=0)
    ideal_E=max((f['E'] for _,f in items),default=1.)
    ideal_R=min((f['R'] for _,f in items),default=0.)
    output=[]
    for i,(c,f) in enumerate(items):
        E,R=f['E'],f['R']
        if family=='O0': value=(c.utility,)
        elif family=='O1': value=(f['axis_E_C']-weight*f['axis_R_G'],)
        elif family=='O2': value=(E-weight*R,)
        elif family=='O3': value=(int(R<=risk_cap),E,-R) if eligible_risk else (-R,E)
        elif family=='O4': value=(-math.floor(R*4),E)
        elif family=='O5': value=(int(E>=efficiency_floor),-R,E) if eligible_eff else (E,-R)
        elif family=='O6': value=(E*(1+f['prompt_age_n'])-weight*R*(1+f['pressure']),)
        elif family=='O7': value=(E-weight*f['tail_R'],)
        elif family=='O8': value=(int(i in pareto),-((ideal_E-E)**2+weight*(R-ideal_R)**2),E)
        elif family=='O9': value=(math.log(max(1e-9,E))-weight*math.log1p(max(0.,R)),)
        elif family=='O10': value=(E/(1+f['yield_fraction'])-weight*R-.01*f['memory_n'],)
        elif family=='O11': value=(E-weight*R-.1*f['uncertainty_n'],)
        elif family=='O12': value=(E-weight*R-.1*(f['switching_C']+f['switching_G']),)
        elif family=='O13':
            gain=math.exp(min(2.,2*(f['feedback_violation']-.10))) if f['feedback_samples']>=32 else 1.
            value=(E-weight*gain*R,)
        output.append(tuple(value))
    return output

def choose(family,items,**kwargs):
    assert items
    ranks=scores(family,items,**kwargs)
    # The same non-perturbing tie rule applies to all action surfaces.
    index=max(range(len(items)),key=lambda i:(ranks[i],items[i][0].complete_tokens_now,
        -items[i][0].residual_chunk_ms,items[i][0].G))
    return items[index][0],ranks[index]
