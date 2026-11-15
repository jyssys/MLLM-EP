"""Reproduce the recorded deadline trap and label fixed-state alternatives."""
import collections
from dataclasses import asdict
from .common import *
from .analyze import records
from .policy import LayerPolicy,Observation


def main():
    name='incremental_guard_550_DIRECT_O16_q4.8_s2026111458_W600_v1'
    folder=ROOT/'raw'/name;assert json.loads((folder/'dispatcher_result.json').read_text())['complete']
    cfg=json.loads((ROOT/'configs'/(name+'.json')).read_text())
    costs=json.loads((ROOT/'configs'/cfg['cost_model']).read_text())
    strict=LayerPolicy(cfg,costs);recovery=LayerPolicy(dict(cfg,expired_ttft_policy='QUEUE_RECOVERY'),costs)
    decisions=records(folder/'online_decisions.jsonl');anchor=decisions[0]['time'];epochs={}
    alternatives=[];mismatches=[]
    for index,row in enumerate(decisions):
        state=Observation(**row['observation']);old=strict.choose(state);new=recovery.choose(state)
        if old.layers!=row['choice']['layers'] or old.reason!=row['choice']['reason']:mismatches.append(index)
        epoch='0_180_s' if row['time']-anchor<180 else '180_360_s' if row['time']-anchor<360 else 'after360_s'
        values=epochs.setdefault(epoch,dict(decisions=0,requests=0,cuts=collections.Counter(),
            reasons=collections.Counter(),overlapping_candidate_gate_counts=collections.Counter()))
        values['decisions']+=1;values['requests']+=state.group_size
        values['cuts'][old.layers]+=state.group_size;values['reasons'][old.reason]+=state.group_size
        for c in old.candidates:values['overlapping_candidate_gate_counts'].update(set(c['rejection_reasons']))
        if old.layers==0 and new.layers>0:
            alternatives.append(dict(index=index,time=row['time'],requests=state.group_size,
                original_reason=old.reason,alternative=asdict(new)))
    assert not mismatches,'Strict default must reproduce the recorded choice'
    write(ROOT/'analysis/expired_TTFT_admission_trap.json',dict(name=name,epochs=epochs,
        exact_choice_replay=True,alternative_requests=sum(x['requests'] for x in alternatives),alternatives=alternatives,
        measured_evidence='TRACE_ANALYSIS',alternative_evidence='OFFLINE_ORACLE',
        scope='Observed complete run and exact strict-choice replay. Alternative choices keep later observed states fixed. They do not predict closed-loop throughput or turn missed SLOs into passes. Only TTFT deadlines already expired at the decision are exempt. Existing ITL and future TTFT predicates remain unchanged. Candidate gates overlap.'))
    print('EXPIRED_TTFT_OFFLINE_ALTERNATIVE_REQUESTS',sum(x['requests'] for x in alternatives),flush=True)


if __name__=='__main__':main()
