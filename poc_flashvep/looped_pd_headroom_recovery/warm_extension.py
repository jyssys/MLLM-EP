"""Minimal exact metadata and supported P-slot controls after graph warm-up."""
import copy
from .common import *
from .initial_campaign import run,gate
from .primary_campaign import subset_parity
from .async_campaign import integrity
from .prepare import put


def main():
    while True:
        p=ROOT/'state/warm_campaign.json'
        if p.exists() and json.loads(p.read_text()).get('stage')=='WARM_EPOCH_COMPLETE_REVIEW_REQUIRED':break
        time.sleep(5)
    from .warm_campaign import make
    plan=make();ledger=[]
    for item in plan['models']:
        accepted=ROOT/'raw'/item['byte_gate']/'recovery_gate.json'
        if not accepted.exists() or not json.loads(accepted.read_text()).get('passed'):
            ledger.append(dict(model=item['model'],status='NOT_RUN_NO_VALID_WARMED_GATE'))
            continue
        source=item['controls'][-1]
        cfg=json.loads((ROOT/'configs'/(source+'.json')).read_text())
        cfg.update(name=f'warm_metadata_unpacked_{item["model"]}_n24',exact_descriptor_packing=False,phase='generic_metadata_control')
        write(ROOT/'configs'/(cfg['name']+'.json'),cfg)
        okay=run(cfg['name']) and subset_parity(source,cfg['name']) and integrity(cfg['name'])
        ledger.append(dict(name=cfg['name'],passed=okay))
    item=next(r for r in plan['models'] if r['model']=='huginn')
    accepted=ROOT/'raw'/item['byte_gate']/'recovery_gate.json'
    if not accepted.exists() or not json.loads(accepted.read_text()).get('passed'):
        write(ROOT/'state/metadata_controls.json',dict(stage='GENERIC_CONTROLS_COMPLETE_REVIEW_REQUIRED',
            time=time.time(),results=ledger,status='HUGINN_EXTENSION_BLOCKED_BY_WARMED_GATE'))
        return
    cfg=json.loads((ROOT/'configs'/(item['byte_gate']+'.json')).read_text())
    cfg.update(name='warm_huginn_supported_Pslots2_byte_gate2',P_slots=2)
    write(ROOT/'configs'/(cfg['name']+'.json'),cfg)
    okay=run(cfg['name']) and gate(cfg['name']) and subset_parity(item['output_gate'],cfg['name']) and integrity(cfg['name'])
    ledger.append(dict(name=cfg['name'],passed=okay))
    if okay:
        base=copy.deepcopy(cfg)
        for chunk in (2560,2048):
            for rate in (.8,1.,1.2):
                c=copy.deepcopy(base)
                c.update(name=f'warm_huginn_Pslots2_c{chunk}_q{str(rate).replace(".","p")}_pilot24',
                    chunk=chunk,phase='qps_pilot',state_probe=False,verify_transfer=False)
                name=put(c,24,rate)
                okay=run(name) and subset_parity(item['output_gate'],name) and integrity(name)
                ledger.append(dict(name=name,passed=okay));write(ROOT/'state/warm_extension_results.json',ledger)
                if not okay:break
            if not okay:break
    write(ROOT/'state/metadata_controls.json',dict(stage='GENERIC_CONTROLS_COMPLETE_REVIEW_REQUIRED',status='WARM_DURABLE_REPLACEMENT_CONTROLS_COMPLETE',
        time=time.time(),results=ledger,scope='Older cold-graph or synchronous-writer matrices remain explicitly NOT_RUN_SUPERSEDED. No old run is relabeled as executed.'))
    restore_if_idle()


if __name__=='__main__':main()
