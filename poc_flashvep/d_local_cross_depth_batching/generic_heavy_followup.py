"""Test prefill pressure only after a generic medium-load decode screen passes."""
import time
from .prepare import *
from .run import execute
from .analyze import requests,pct


def safe(name,model):
    folder=ROOT/'raw'/name;receipt=folder/'dispatcher_result.json'
    if not receipt.exists():return None
    if not json.loads(receipt.read_text()).get('complete'):return False
    rows=requests(folder)
    if len(rows)!=128:return False
    from .qualification import compare_outputs
    if not compare_outputs(f'local_{model}_native_n128_s2026110701_off',name,128)['passed']:return False
    slos=json.loads((ROOT/'configs/frozen_SLO.json').read_text())[f'{model}_L{4096 if model=="ouro" else 2048}']
    itl=[1000*(e['time']-e['previous_time']) for r in rows for e in r['token_events'] if not e['first'] and e['new_tokens']==1]
    return pct([r['tpot_ms'] for r in rows],95)<=.9*slos['TPOT_ms'] and pct(itl,99)<=slos['ITL_ms']


def main():
    initialize()
    for model in ('ouro','huginn'):
        first=f'serve_{model}_paced_c128_fixed25_v5';credit=f'serve_{model}_paced_credit_c128_fixed25_v1'
        chosen=None
        while time.time()<LONG_START_CUTOFF:
            prior=safe(first,model)
            if prior is True:chosen=first;break
            if prior is False:
                receipt=json.loads((ROOT/'raw'/first/'dispatcher_result.json').read_text())
                parity=ROOT/'raw'/first/'serving_output_parity.json'
                if not receipt.get('complete') or (parity.exists() and not json.loads(parity.read_text())['passed']):break
                value=safe(credit,model)
                if value is True:chosen=credit;break
                if value is False:break
                rejected=False
                for count in (8,32):
                    gate=ROOT/'raw'/f'credit_{model}_mixed_state{count}_v1/local_qualification.json'
                    if gate.exists() and not json.loads(gate.read_text())['passed']:rejected=True
                    failed=gate.parent/'dispatcher_result.json'
                    if failed.exists() and not json.loads(failed.read_text()).get('complete'):rejected=True
                if rejected:break
            time.sleep(10)
        write(ROOT/'state'/f'{model}_generic_heavy_selection.json',dict(time=time.time(),chosen=chosen,
            rule='Prefer qualified loop-paced separate baseline. Use output-credit control only if the first screen fails. No performance-based selection.'))
        if chosen is None:continue
        cfg=configuration(model,'paced',rate=6.356 if model=='ouro' else 1.971)
        cfg.update(name=f'serve_{model}_paced_heavy_fixed25_v1',dlocal_chunk=128,dlocal_every=4,
            dlocal_max_active=1,dlocal_output_credit='credit_' in chosen,
            deep_complete_D_geometry_warmup=True,load='heavy',source_medium_control=chosen)
        save(cfg);execute(cfg['name'])


if __name__=='__main__':main()
