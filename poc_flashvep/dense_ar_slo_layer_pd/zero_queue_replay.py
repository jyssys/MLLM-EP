"""Test whether a queue-only early gate hides modeled personal benefits."""
from dataclasses import fields
from .common import *
from .analyze import records
from .policy import LayerPolicy,Observation


def main():
    names=['method2_DIRECT_G8_D8_wait0_L0_O16_q4.8_N128_v1',
        'method2_DIRECT_G8_D8_wait1_L0_O16_q4.8_N128_v1']
    result=[];known={f.name for f in fields(Observation)}
    for name in names:
        cfg=json.loads((ROOT/'configs'/(name+'.json')).read_text())
        cfg.update(separate_nominal_objective=True)
        model=json.loads((ROOT/'configs/coalesced_cost_model_v2.json').read_text())
        for row in records(ROOT/'raw'/name/'online_decisions.jsonl'):
            s=Observation(**{k:v for k,v in row['observation'].items() if k in known})
            if s.P_backlog:continue
            original=LayerPolicy(cfg,model).choose(s)
            candidate=LayerPolicy(dict(cfg,allow_zero_queue_personal_benefit=True),model).choose(s)
            result.append(dict(name=name,group_size=s.group_size,P_actual_observed_backlog=s.P_backlog,
                original=original.layers,counterfactual=candidate.layers,reason=candidate.reason,
                selected_candidate=next((c for c in candidate.candidates if c['layers']==candidate.layers),None)))
    accepted=[r for r in result if r['counterfactual']]
    write(ROOT/'analysis/zero_queue_personal_benefit_replay.json',dict(rows=result,
        evaluated_groups=len(result),accepted_groups=len(accepted),accepted_requests=sum(r['group_size'] for r in accepted),
        evidence='OFFLINE_ORACLE',scope='Original observed states remain fixed. The alternative requires positive modeled personal completion benefit and all existing safety predicates. It does not invent queue pressure or observe future arrivals. No downstream-state replay or online improvement is established.'))
    print('ZERO_QUEUE_COUNTERFACTUAL',len(result),len(accepted),sum(r['group_size'] for r in accepted),flush=True)


if __name__=='__main__':main()
