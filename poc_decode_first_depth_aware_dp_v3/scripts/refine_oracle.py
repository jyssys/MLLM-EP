"""Stronger bounded clairvoyant search. Preserve the first search as v1 evidence."""
import random
import numpy as np
from poc_decode_first_depth_aware_dp_v3.contract import *
from poc_decode_first_depth_aware_dp_v3.simulator import CostModel,simulate_dp
from poc_decode_first_depth_aware_dp_v3.scripts.oracle import packed

def search(inputs,model,starts):
    cache={}
    def objective(p):
        key=tuple(p)
        if key not in cache:cache[key]=simulate_dp(packed(inputs,p),model)['E2E_P95_ms']
        return cache[key]
    best=list(min(starts,key=objective));score=objective(best)
    rng=random.Random(20261018)
    for sweep in range(3):
        improved=False
        # A full one-request-move neighborhood, not 20 random trials.
        candidates=[]
        for i in range(len(best)):
            for rank in range(4):
                if rank==best[i]:continue
                trial=list(best);trial[i]=rank;candidates.append(trial)
        # Pair swaps retain each rank's request count.
        pairs=list(__import__('itertools').combinations(range(len(best)),2))
        rng.shuffle(pairs)
        for i,j in pairs[:min(128,len(pairs))]:
            if best[i]==best[j]:continue
            trial=list(best);trial[i],trial[j]=trial[j],trial[i];candidates.append(trial)
        for trial in candidates:
            value=objective(trial)
            if value<score-1e-6:best=trial;score=value;improved=True
        print('ORACLE_REFINEMENT',len(inputs),sweep,len(cache),score,flush=True)
        if not improved:break
    return best,simulate_dp(packed(inputs,best),model),len(cache)

def main():
    fit=read(ROOT/'artifacts/cdb_cost_model.json')
    assert fit['status']=='VALIDATED'
    model=CostModel(fit['parameters'])
    frozen=read(ROOT/'artifacts/discovery_frozen.json')['rows']
    heldout=[r for r in frozen if r['split']=='heldout']
    old=read(ROOT/'artifacts/oracle_placements.json')
    oldmetrics=read(ROOT/'artifacts/oracle_schedule_results.json')
    results=[]
    placements=[]
    for c in [8,16,32,64]:
        inputs=[]
        for order,r in enumerate(heldout[:c]):
            raw=read(ROOT/'outputs/discovery128_graph_b1_v1'/(r['id'].replace('/','_')+'.json'))
            inputs.append(dict(id=r['id'],prompt_tokens=raw['prompt_tokens'],depths=[e['depth'] for e in raw['depth_events']],
                arrival_ms=0.,order=order,cap=256))
        starts=[[x['rank'] for x in record['placements']] for record in old if record['C']==c]
        selected,best,evaluations=search(inputs,model,starts)
        candidates=[r for r in oldmetrics if r['C']==c and r['policy'] in ['round_robin','queue','length_causal','strong_cost','true_length_best_known']]
        control=min(candidates,key=lambda r:r['E2E_P95_ms'])
        control_place=next([x['rank'] for x in r['placements']] for r in old if r['C']==c and r['policy']==control['policy'])
        baseline=simulate_dp(packed(inputs,control_place),model)
        byid=lambda data:{r['id']:r['finish_ms']-r['arrival_ms'] for r in data['outputs']}
        a,b=byid(baseline),byid(best)
        ids=sorted(a)
        av=np.asarray([a[i] for i in ids]);bv=np.asarray([b[i] for i in ids])
        draws=np.random.default_rng(20261018).integers(0,c,(2000,c))
        gains=100*(1-np.quantile(bv[draws],.95,axis=1)/np.quantile(av[draws],.95,axis=1))
        result=dict(C=c,control=control['policy'],control_E2E_P95_ms=baseline['E2E_P95_ms'],
            oracle_E2E_P95_ms=best['E2E_P95_ms'],gain_percent=100*(1-best['E2E_P95_ms']/baseline['E2E_P95_ms']),
            CI95_gain_percent=np.quantile(gains,[.025,.975]).tolist(),evaluations=evaluations,
            model_makespan_ms=best['makespan_ms'],model_requests_per_s=best['requests_per_s'],
            label='VALIDATED_COST_BEST_KNOWN_CLAIRVOYANT_SEARCH_NOT_PHYSICAL_SPEEDUP',
            optimality='Three full move neighborhoods plus up to 128 swaps per sweep. Not a proven global optimum.',
            comparison='Best observed non-depth control, including the true-length heuristic. Do not compare only against a worse length heuristic.')
        results.append(result)
        placements.append(dict(C=c,placements=[dict(id=r['id'],rank=rank) for r,rank in zip(inputs,selected)]))
        write(ROOT/'artifacts/oracle_refinement_v2.json',dict(rows=results,placements=placements,
            bootstrap_unit='Paired requests, conditional on fixed selected placements. No reoptimization in bootstrap.',
            predictor_allowed=any(r['gain_percent']>=10 and r['CI95_gain_percent'][0]>0 for r in results)))
        print('REFINED_RESULT',result,flush=True)

if __name__=='__main__':main()
