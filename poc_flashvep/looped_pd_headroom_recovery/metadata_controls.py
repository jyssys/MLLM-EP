"""Controlled generic host-overhead and supported P-slot diagnostics."""
import copy
from .common import *
from .prepare import put
from .initial_campaign import run,gate
from .primary_campaign import subset_parity


def make():
    revised=ROOT/'state/metadata_controls_plan_v2.json'
    if revised.exists():return json.loads(revised.read_text())
    target=ROOT/'state/metadata_controls_plan.json'
    if target.exists():return json.loads(target.read_text())
    sources=dict(ouro='primary_ouro_l6144_o64_c6144_gate8',huginn='primary_huginn_l2560_o32_c2560_gate8')
    result=[]
    for model,source in sources.items():
        base=json.loads((ROOT/'configs'/(source+'.json')).read_text())
        base.update(timeout=900,workload=f'L{base["length"]}_O{base["cap"]}',nsys=False,
            P_only_profiler=False,observe_descriptor_cost=True,record_descriptor_replay=True)
        for key in ('capacity_reference','reference_capacity','rate_factor','rate_scope'):base.pop(key,None)
        variants=[('legacy_observed',{}),('packed',dict(exact_descriptor_packing=True))]
        if model=='ouro':
            limits=dict(base.get('native_transport_limits',{}),max_inflight_bytes=8*1024**3)
            variants.extend([('credit8g',dict(native_transport_limits=limits)),
                ('credit8g_packed',dict(native_transport_limits=limits,exact_descriptor_packing=True))])
        if model=='huginn':
            variants=[('legacy_observed',{}),('Pslots2',dict(P_slots=2)),('Pslots2_packed',dict(P_slots=2,exact_descriptor_packing=True))]
        for label,delta in variants:
            cfg=copy.deepcopy(base)
            cfg.update(name=f'generic_control_{model}_{label}_gate8',phase='correctness_generic_control',**delta)
            check=put(cfg,8)
            pilots=[]
            for rate in ([2.,3.] if model=='ouro' else [.8,1.2]):
                cfg=copy.deepcopy(cfg)
                cfg.update(name=f'generic_control_{model}_{label}_q{str(rate).replace(".","p")}_pilot24',phase='qps_pilot',
                    verify_transfer=False,state_probe=False)
                pilots.append(put(cfg,24,rate))
            result.append(dict(model=model,label=label,reference=source,gate=check,pilots=pilots))
    write(target,dict(created=time.time(),items=result,
        scope='Optional generic diagnostics. Exact descriptor packing changes CPU tuple construction only. Pslots2 uses supported native batching. Never attribute these changes to loop-specific novelty.'))
    return json.loads(target.read_text())


def main():
    while True:
        path=ROOT/'state/followup_campaign.json'
        state=json.loads(path.read_text()) if path.exists() else {}
        if state.get('stage')=='FOLLOWUP_COMPLETE_REVIEW_REQUIRED':break
        time.sleep(5)
    ledger=[]
    for item in make()['items']:
        reference=ROOT/'raw'/item['reference']/'recovery_gate.json'
        if not reference.exists() or not json.loads(reference.read_text())['passed']:continue
        write(ROOT/'state/metadata_controls.json',dict(stage='GENERIC_CONTROL_CORRECTNESS',item=item,time=time.time()))
        okay=run(item['gate']) and gate(item['gate'],item['reference'])
        ledger.append(dict(name=item['gate'],passed=okay))
        if okay:
            for name in item['pilots']:
                okay=run(name) and subset_parity(item['reference'],name)
                ledger.append(dict(name=name,passed=okay))
                if not okay:break
        write(ROOT/'state/metadata_controls_results.json',ledger)
    write(ROOT/'state/metadata_controls.json',dict(stage='GENERIC_CONTROLS_COMPLETE_REVIEW_REQUIRED',time=time.time(),results=ledger))
    restore_if_idle()


if __name__=='__main__':main()
