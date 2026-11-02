"""Test whether source retention hides compute behind a supported staging limit."""
import copy
from .common import *
from .prepare import put
from .initial_campaign import run,gate
from .gates import one


def main():
    write(ROOT/'state/source_credit_control.json',dict(stage='WAITING_FOR_CONFIRMATION',time=time.time(),
        hypothesis='If ACK-held source slots restrict submission, four supported extra draining slots can increase P submission/service without early KV release.',
        alternative='Descriptor preparation, wire progress, or D reservation can remain binding. A source-slot change need not improve compute occupancy.',
        scope='Post-pilot causal diagnostic. It does not alter the frozen N128 PD/DP comparison or count as an independent repeat.'))
    while True:
        p=ROOT/'state/confirmation.json'
        if p.exists() and json.loads(p.read_text()).get('stage')=='CONFIRMATION_COMPLETE_REVIEW_REQUIRED':break
        time.sleep(5)
    selection=next(r for r in json.loads((ROOT/'state/confirmation_plan.json').read_text())['selection'] if r['model']=='ouro')
    source=json.loads((ROOT/'configs'/(selection['PD_pilot']+'.json')).read_text())
    cfg=copy.deepcopy(source)
    cfg.update(name='source_credit_ouro_l8192_extra4_gate8',phase='correctness',verify_transfer=True,state_probe=True,
        timeout=600,source_credit_diagnostic=True)
    cfg['native_transport_limits']=dict(cfg['native_transport_limits'],max_draining_requests=4)
    put(cfg,8,2.5)
    write(ROOT/'state/source_credit_control.json',dict(stage='BYTE_AND_OUTPUT_GATE',time=time.time(),name=cfg['name']))
    valid=run(cfg['name']) and gate(cfg['name'])
    results=[]
    if valid:
        from .primary_campaign import subset_parity
        valid=subset_parity('warm_secondary_ouro_l8192_output_gate8',cfg['name'])
    if valid:
        for extra in (0,4):
            c=copy.deepcopy(source)
            c.update(name=f'source_credit_ouro_l8192_extra{extra}_q2p5_pilot24',phase='source_credit_diagnostic',
                verify_transfer=False,state_probe=False,timeout=600,source_credit_diagnostic=True)
            c['native_transport_limits']=dict(c['native_transport_limits'],max_draining_requests=extra)
            put(c,24,2.5)
            write(ROOT/'state/source_credit_control.json',dict(stage='MATCHED_PILOT',time=time.time(),name=c['name']))
            okay=run(c['name'])
            result=dict(name=c['name'],complete=okay,extra_draining_slots=extra)
            if okay:result['same_start_gates']=one(c)
            results.append(result)
    write(ROOT/'state/source_credit_control.json',dict(stage='SOURCE_CREDIT_CONTROL_COMPLETE_REVIEW_REQUIRED',time=time.time(),
        gate=valid,results=results,scope='Supported source staging capacity only. No ACK bypass, no model change, no claim of sustainable capacity or a Loop-specific method.'))


if __name__=='__main__':main()
