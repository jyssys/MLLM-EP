"""Retain minimal metadata/slot controls in the valid durable-evidence epoch."""
import copy
from .common import *
from .initial_campaign import run,gate
from .primary_campaign import subset_parity
from .async_campaign import integrity


def main():
    while True:
        path=ROOT/'state/async_campaign.json'
        if path.exists() and json.loads(path.read_text()).get('stage')=='ASYNC_EPOCH_COMPLETE_REVIEW_REQUIRED':break
        time.sleep(5)
    plan=json.loads((ROOT/'state/async_epoch_plan.json').read_text())
    ledger=[]
    for item in plan['models']:
        source=json.loads((ROOT/'configs'/(item['controls'][1]+'.json')).read_text())
        cfg=copy.deepcopy(source)
        cfg.update(name=f'async_metadata_unpacked_{item["model"]}_n24',exact_descriptor_packing=False,
            phase='generic_metadata_control')
        target=ROOT/'configs'/(cfg['name']+'.json')
        if target.exists():assert json.loads(target.read_text())==cfg
        else:write(target,cfg)
        okay=run(cfg['name']) and subset_parity(item['controls'][1],cfg['name']) and integrity(cfg['name'])
        ledger.append(dict(name=cfg['name'],passed=okay))
    h=next(r for r in plan['models'] if r['model']=='huginn')
    cfg=json.loads((ROOT/'configs'/(h['native_gate']+'.json')).read_text())
    cfg.update(name='async_huginn_supported_Pslots2_gate8',P_slots=2)
    target=ROOT/'configs'/(cfg['name']+'.json')
    if target.exists():assert json.loads(target.read_text())==cfg
    else:write(target,cfg)
    okay=run(cfg['name']) and gate(cfg['name'],h['native_gate']) and integrity(cfg['name'])
    ledger.append(dict(name=cfg['name'],passed=okay))
    if okay:
        from .prepare import put
        base=copy.deepcopy(cfg)
        for chunk in (2560,2048):
            for rate in (.8,1.,1.2):
                cfg=copy.deepcopy(base)
                cfg.update(name=f'async_huginn_Pslots2_c{chunk}_q{str(rate).replace(".","p")}_pilot24',
                    chunk=chunk,phase='qps_pilot',state_probe=False,verify_transfer=False)
                name=put(cfg,24,rate)
                okay=run(name) and subset_parity(h['native_gate'],name) and integrity(name)
                ledger.append(dict(name=name,passed=okay))
                write(ROOT/'state/async_extension_results.json',ledger)
                if not okay:break
            if not okay:break
    write(ROOT/'state/metadata_controls.json',dict(stage='GENERIC_CONTROLS_COMPLETE_REVIEW_REQUIRED',time=time.time(),
        status='ASYNC_EPOCH_CONTROLS_COMPLETE',results=ledger,
        superseded_original_plan='metadata_controls_plan_v2.json',
        scope='Original synchronous-writer controls remain NOT_RUN_SUPERSEDED. The replacement controls retain exact metadata A/B and a supported H P-slot control under the same durable writer used by both serving baselines.'))
    restore_if_idle()


if __name__=='__main__':main()
