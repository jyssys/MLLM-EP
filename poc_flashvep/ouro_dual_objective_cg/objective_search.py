"""Historical development survey. Scores are not alternative GPU outcomes."""
import csv
import gzip
import json
from collections import Counter,defaultdict
import numpy as np
from .common import ROOT,PRIOR,write,digest
from .objective_features import prepare_items,contextualize_offline
from .objective_families import choose,NAMES
from ..ouro_budget_joint_cg.cost_predictor import ServiceLookup,csv_file
from ..ouro_budget_joint_cg.optimizer import enumerate_candidates

def calibration(model):
    """Only old TRAIN data determine normalization and service-tail factors."""
    observations=defaultdict(list)
    with gzip.open(PRIOR/'cost_service_raw.csv.gz','rt') as stream:
        for row in csv.DictReader(stream):
            if row['split']=='TRAIN' and int(row['prompt_rows'])==1 and int(row['decode_rows'])==0:
                observations[f"{row['path']}:{row['G']}"].append(float(row['service_ms']))
    tail={key:float(np.mean([x for x in values if x>=np.percentile(values,95)])/max(.001,np.median(values)))
        for key,values in observations.items() if len(values)>=30}
    rates=[];exposures=[]
    for P in (256,512,1800,2048,4096,8192):
        state=dict(kind='NEW_CHUNK',C=0,start=0,depth=0,prompt_remaining=P,token_budget=2048,atomic=False)
        for c in enumerate_candidates(state,float('inf'),model):
            rates.append(c.C/c.residual_chunk_ms);exposures.append(max(c.current_ms,c.opening_ms))
    return dict(version='DUAL_NORMALIZATION_V1',efficiency_scale=float(np.percentile(rates,95)),risk_scale=1.,
        service_scale_ms=float(np.percentile(exposures,95)),reference_TBT_ms=200.,submission_cost_ms=.1,
        submission_cost_scope='Explicit 0.1-ms sensitivity prior, not a measured isolated marginal launch cost.',
        tail_ratio=tail,tail_scope='Old TRAIN conditional service-tail proxy. Not conditional future ITL probability.',
        source_sha256=digest(PRIOR/'cost_service_raw.csv.gz'),
        epsilon_ms=.001,E_clip='No clipping',R_clip='No clipping',
        role='Historical TRAIN development calibration. Newly collected source families remain separate.')

def main():
    model=ServiceLookup(json.loads((PRIOR/'cost_model_artifacts/service_lookup_v1.json').read_text()))
    norm=calibration(model);target=ROOT/'objective_normalization_v1.json'
    if target.exists(): assert json.loads(target.read_text())==norm,'Frozen normalization changed'
    else: write(target,norm)
    snapshots=[]
    # Only historical TRAIN pilot states select the development shortlist.
    # Consumed previous FINAL is retained separately for the mandatory Phase A.
    for folder in sorted((PRIOR/'raw').glob('cb_p3_budget_*')):
        for file in sorted(folder.glob('J1_T*.json')):
            data=json.loads(file.read_text())
            for i,d in enumerate(data['optimizer_decisions']):
                s=d['state']
                if s['kind']!='NEW_CHUNK' or not s['request_event_ready']: continue
                state=contextualize_offline(s,data,d['host_s'])
                snapshots.append((folder.name,data['trial'],i,data['P'],state,d['budget_ms']))
    # Avoid thousands of copies of the same source/frontier geometry.
    dedup={}
    for row in snapshots:
        run,trial,index,P,s,B=row
        key=(run,trial,s['id'],s['start'],B)
        dedup.setdefault(key,row)
    snapshots=list(dedup.values())
    results=[];transitions=[];gradients=[];fixed_regret=defaultdict(list)
    variants=[(family,lam) for family in NAMES for lam in (0.,.1,.25,.5,1.,2.,4.,8.)]
    counts=defaultdict(Counter)
    for run,trial,index,P,state,B in snapshots:
        candidates=enumerate_candidates(state,B,model)
        if not candidates: continue
        items=prepare_items(candidates,state,norm,model)
        # Fixed-C per-family tuning is performed over the same training states below.
        E,_=choose('O2',items,endpoint='E_ONLY');R,_=choose('O2',items,endpoint='R_ONLY')
        for family,lam in variants:
            chosen,rank=choose(family,items,weight=lam)
            counter=counts[(family,lam)]
            counter['states']+=1;counter[f'C{chosen.C}']+=1;counter[f'G{chosen.G}']+=1
            counter['C_ne_efficiency']+=chosen.C!=E.C;counter['G_ne_risk']+=chosen.G!=R.G
            counter['multi_C']+=len({c.C for c in candidates})>1
            counter['both_ne_efficiency']+=chosen.C!=E.C and chosen.G!=E.G
            for fixed_C in (64,128,256,512,768,1024,1536,2048):
                actual=min(fixed_C,state['prompt_remaining'],state['token_budget'])
                restricted=[pair for pair in items if pair[0].C==actual]
                if restricted:
                    alternative,alternative_rank=choose(family,restricted,weight=lam)
                    # Ordinal training regret avoids averaging incompatible
                    # lexicographic score dimensions as though they were ms.
                    all_ranks=[]
                    from .objective_families import scores
                    all_ranks=scores(family,items,weight=lam)
                    regret=sum(r>alternative_rank for r in all_ranks)/len(all_ranks)
                else:
                    regret=1.
                fixed_regret[(family,lam,fixed_C)].append(regret)
            transitions.append(dict(run_id=run,trial=trial,decision=index,P=P,B_ms=B,
                objective=family,weight=lam,chosen_C=chosen.C,chosen_G=chosen.G,E_only_C=E.C,E_only_G=E.G,
                R_only_C=R.C,R_only_G=R.G,score=json.dumps(rank),known=chosen.known,
                evidence='OFFLINE_ESTIMATE_ON_CONSUMED_HISTORICAL_TRAIN_STATES'))
        lookup={(c.C,c.G):(c,f) for c,f in items}
        cs=sorted({c.C for c in candidates})
        if len(cs)>1:
            low,high=cs[0],cs[-1]
            for g in (1,2,3,4):
                if (low,g) not in lookup or (high,g) not in lookup: continue
                a,b=lookup[(low,g)][1],lookup[(high,g)][1]
                gradients.append(dict(run_id=run,trial=trial,P=P,B_ms=B,axis='C',fixed=g,low=low,high=high,
                    delta_E=b['E']-a['E'],delta_R=b['R']-a['R'],
                    both_known=lookup[(low,g)][0].known and lookup[(high,g)][0].known,
                    evidence='OFFLINE_FROZEN_COST_FEATURE_DIFFERENCE_NOT_LIVE_COUNTERFACTUAL'))
            for c in cs:
                available=sorted(g for x,g in lookup if x==c)
                if len(available)<2: continue
                lowg,highg=available[0],available[-1];a,b=lookup[(c,lowg)][1],lookup[(c,highg)][1]
                gradients.append(dict(run_id=run,trial=trial,P=P,B_ms=B,axis='G',fixed=c,low=lowg,high=highg,
                    delta_E=b['E']-a['E'],delta_R=b['R']-a['R'],
                    both_known=lookup[(c,lowg)][0].known and lookup[(c,highg)][0].known,
                    evidence='OFFLINE_FROZEN_COST_FEATURE_DIFFERENCE_NOT_LIVE_COUNTERFACTUAL'))
    for (family,lam),count in counts.items():
        states=count['states'];multi=count['multi_C']
        results.append(dict(objective=family,name=NAMES[family],weight=lam,states=states,multi_C_states=multi,
            infeasible_fresh_states=len(snapshots)-states,
            C_differs_E_only=count['C_ne_efficiency'],C_differs_E_only_fraction=count['C_ne_efficiency']/max(1,multi),
            G_differs_R_only=count['G_ne_risk'],both_differ_E_only=count['both_ne_efficiency'],
            C_histogram=json.dumps({k:v for k,v in count.items() if k.startswith('C') and k[1:].isdigit()}),
            G_histogram=json.dumps({k:v for k,v in count.items() if k.startswith('G') and k[1:].isdigit()}),
            interpretation='Action-score geometry only. No serving benefit measured.'))
    csv_file(ROOT/'objective_grid.csv',results)
    csv_file(ROOT/'objective_action_transitions.csv.gz',transitions)
    csv_file(ROOT/'objective_gradients.csv',gradients)
    tuning=[dict(objective=f,weight=w,fixed_C=c,states=len(v),mean_ordinal_regret=float(np.mean(v)),
        scope='Historical TRAIN objective regret, not measured serving cost') for (f,w,c),v in fixed_regret.items()]
    csv_file(ROOT/'fixed_C_objective_tuning.csv',tuning)
    write(ROOT/'offline_search_summary.json',dict(historical_TRAIN_fresh_states=len(snapshots),
        variants=len(results),model_sha256=digest(PRIOR/'cost_model_artifacts/service_lookup_v1.json'),
        normalization_sha256=digest(target),future_complete_token_predictor=False,
        live_shortlist='PENDING fresh VALIDATION serving. No FINAL outcome used.'))
    print('Offline family variants',len(results),'fresh historical TRAIN states',len(snapshots),
        'score transitions',len(transitions),'axis differences',len(gradients))

if __name__=='__main__': main()
