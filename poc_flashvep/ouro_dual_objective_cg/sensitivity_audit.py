"""Factorial feature sensitivity on constructed states, not live traffic.

The feature/cost functions and finalist weights remain frozen.
Constructed rows do not contribute to a serving-gain or promotion gate.
"""
import json
from .common import ROOT,PRIOR,write
from .objective_features import prepare_items
from .objective_families import choose
from ..ouro_budget_joint_cg.optimizer import enumerate_candidates
from ..ouro_budget_joint_cg.cost_predictor import ServiceLookup,csv_file

def main():
    model=ServiceLookup(json.loads((PRIOR/'cost_model_artifacts/service_lookup_v1.json').read_text()))
    norm=json.loads((ROOT/'objective_normalization_v1.json').read_text());rows=[]
    for P in (256,512,1800,2048,4096,8192):
        for B in (20,40,80,120,240):
            for n in (0,1,4,8,16):
                for age in (0,100,500):
                    if n==0 and age:continue
                    for prompt_age in (0,4000):
                        state=dict(id='CONSTRUCTED',kind='NEW_CHUNK',phase='prefilling_prelude',C=0,start=0,
                            depth=0,prompt_remaining=P,token_budget=2048,atomic=False,request_event_ready=True,
                            issued_decode=[dict(id=str(i)) for i in range(n)],decode_age_max_ms=age,
                            prompt_age_ms=prompt_age,TBT_target_ms=200,past_TBT_samples=0)
                        items=prepare_items(enumerate_candidates(state,B,model),state,norm,model)
                        base=dict(P=P,B_ms=B,decode_count=n,delivered_gap_age_ms=age,prompt_age_ms=prompt_age,
                            phase='fresh chunk',candidate_count=len(items),
                            evidence='SIMULATION_ONLY_CONSTRUCTED_STATE_FEATURE_RESPONSE_NOT_GPU_MEASUREMENT',
                            cost_scope='One-prefill-span historical lookup. Does not model active-decode contention or C16 batch service.')
                        if not items:
                            rows.append(dict(base,objective='ALL',status='NO_BUDGET_FEASIBLE_ACTION'));continue
                        e,_=choose('O2',items,endpoint='E_ONLY');r,_=choose('O2',items,endpoint='R_ONLY')
                        for family,weight in [('O0',1),('O1',2),('O2',8),('O3',1),('O4',1),('O5',1),('O6',2),('O7',2),('O8',2),('O9',2),('O10',8),('O11',2),('O12',2),('O13',2)]:
                            c,_=choose(family,items,weight=weight)
                            rows.append(dict(base,objective=family,weight=weight,status='SCORED',C=c.C,G=c.G,
                                E_only_C=e.C,E_only_G=e.G,R_only_C=r.C,R_only_G=r.G,
                                different_actual_C_from_fixed=c.C!=min(2048,P),known_shape=c.known,
                                both_axes_differ_E=c.C!=e.C and c.G!=e.G))
    keys=list(dict.fromkeys(k for row in rows for k in row))
    csv_file(ROOT/'constructed_state_sensitivity.csv.gz',[{k:r.get(k) for k in keys} for r in rows])
    write(ROOT/'sensitivity_scope.json',dict(rows=len(rows),uses_future_GPU_time=False,
        serving_measurements=False,changes_finalist=False,missing='Future complete-token ITL, graph contention, actual batch packing, and empirical probability labels.'))
    print('Constructed-state sensitivity rows',len(rows))

if __name__=='__main__':main()
