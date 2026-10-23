"""TRAIN-only constraint-threshold sensitivity. No online retuning."""
import json
from collections import Counter
import numpy as np
from .common import ROOT,PRIOR,write
from .objective_search import contextualize_offline
from .objective_features import prepare_items
from .objective_families import choose
from ..ouro_budget_joint_cg.optimizer import enumerate_candidates
from ..ouro_budget_joint_cg.cost_predictor import ServiceLookup,csv_file

def main():
    model=ServiceLookup(json.loads((PRIOR/'cost_model_artifacts/service_lookup_v1.json').read_text()))
    norm=json.loads((ROOT/'objective_normalization_v1.json').read_text());states={}
    for folder in sorted((PRIOR/'raw').glob('cb_p3_budget_*')):
        for file in sorted(folder.glob('J1_T*.json')):
            data=json.loads(file.read_text())
            for d in data['optimizer_decisions']:
                s=d['state']
                if s['kind']!='NEW_CHUNK' or not s['request_event_ready']:continue
                key=(folder.name,data['trial'],s['id'],s['start'],d['budget_ms'])
                if key in states:continue
                states[key]=None
                state=contextualize_offline(s,data,d['host_s'])
                items=prepare_items(enumerate_candidates(state,d['budget_ms'],model),state,norm,model)
                if items:states[key]=items
    states={k:v for k,v in states.items() if v}
    R=[f['R'] for items in states.values() for c,f in items];E=[f['E'] for items in states.values() for c,f in items]
    rows=[]
    for family,field,values in [('O3','risk_cap',R),('O5','efficiency_floor',E)]:
        for p in (10,25,50,75,90):
            threshold=float(np.percentile(values,p));hist=Counter();different=0;no_feasible_constraint=0
            for items in states.values():
                a,_=choose(family,items,**{field:threshold})
                base,_=choose(family,items,**{field:.5});hist[(a.C,a.G)]+=1;different+=(a.C,a.G)!=(base.C,base.G)
                no_feasible_constraint+=not any(f['R']<=threshold if family=='O3' else f['E']>=threshold for c,f in items)
            rows.append(dict(objective=family,threshold_field=field,TRAIN_percentile=p,threshold=threshold,
                states=len(states),different_actions_vs_original_half=different,
                no_constraint_feasible_states=no_feasible_constraint,
                C_G_histogram=json.dumps({f'C{c}/G{g}':n for (c,g),n in sorted(hist.items())}),
                evidence='OFFLINE_ESTIMATE_CONSUMED_TRAIN. Additional threshold characterization; no FINAL policy or online weight changes.'))
    csv_file(ROOT/'constraint_threshold_sensitivity.csv',rows)
    write(ROOT/'constraint_threshold_scope.json',dict(TRAIN_feasible_states=len(states),TRAIN_candidate_features=len(R),
        final_outcomes_used=False,original_online_shortlist_unchanged=True,
        scope='Offline-only supplementary analysis. Not a prespecified online constraint-family comparison.'))
    print('Constraint sensitivity settings',len(rows))

if __name__=='__main__':main()
