"""Bound prefill submissions if strict ready priority still harms decode."""
import time
from .prepare import *
from .run import execute
from .qualification import compare
from .analyze import requests,pct


def main():
    initialize()
    for model in ('ouro','huginn'):
        previous=f'serve_{model}_paced_c128_fixed25_v5'
        receipt=ROOT/'raw'/previous/'dispatcher_result.json'
        while not receipt.exists():
            state=ROOT/'state/fair_generic_complete.json'
            if state.exists():break
            if time.time()>=LONG_START_CUTOFF:return
            time.sleep(10)
        if not receipt.exists() or not json.loads(receipt.read_text()).get('complete'):continue
        rows=requests(receipt.parent)
        slos=json.loads((ROOT/'configs/frozen_SLO.json').read_text())[f'{model}_L{4096 if model=="ouro" else 2048}']
        itl=[1000*(e['time']-e['previous_time']) for r in rows for e in r['token_events'] if not e['first'] and e['new_tokens']==1]
        safe=pct([r['tpot_ms'] for r in rows],95)<=.9*slos['TPOT_ms'] and pct(itl,99)<=slos['ITL_ms']
        write(ROOT/'state'/f'{model}_credit_followup_decision.json',dict(time=time.time(),prior=previous,prior_safe=safe,
            action='NOT_NEEDED' if safe else 'QUALIFY_ONE_PREFILL_SUBMISSION_PER_OBSERVED_OUTPUT_STEP',
            scope='Reuse output-credit throttle. No full dynamic controller or hard-deadline guarantee. Same 25% fixed placement.'))
        if safe:continue
        valid=True
        for count in (8,32):
            if not valid:break
            cfg=configuration(model,'paced',count,verify=True)
            cfg.update(name=f'credit_{model}_mixed_state{count}_v1',arrival_offsets=[0.]*count,
                dlocal_chunk=128,dlocal_all=False,dlocal_every=4,dlocal_max_active=1,
                dlocal_output_credit=True,prime_all_graph_shapes=False,deep_serial_primary_warmup=False)
            save(cfg)
            reference=f'local_{model}_native_n8_s2026110701_off_state' if count==8 else f'qual32_{model}_native_v3'
            valid=execute(cfg['name']) and compare(reference,cfg['name'],count)['passed']
        if valid:
            cfg=configuration(model,'paced')
            cfg.update(name=f'serve_{model}_paced_credit_c128_fixed25_v1',dlocal_chunk=128,dlocal_every=4,
                dlocal_max_active=1,dlocal_output_credit=True,deep_complete_D_geometry_warmup=True)
            save(cfg);execute(cfg['name'])


if __name__=='__main__':main()
