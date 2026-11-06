"""Qualify larger chunks before marker-free generic-placement controls."""
import time
from .prepare import *
from .run import execute
from .qualification import compare


def dispatch(cfg,stage):
    save(cfg)
    write(ROOT/'state/controller_v4.json',dict(stage=stage,name=cfg['name'],time=time.time()))
    return execute(cfg['name'])


def main():
    initialize()
    write(ROOT/'state/chunk128_selection.json',dict(time=time.time(),chunk=128,
        reason='Measured core enclosure changes little from C16 to C128. Larger chunks reduce launch count. Selection precedes all C128 serving results.',
        placement='Fixed request indices 0,4,...,124 in both generic arms. Local active limit 1 for both. Not a complete deadline-prediction policy.',
        Phase_D='Cross-depth online extension remains NOT_RUN_PHASE_C_GATE. These are Phase B generic controls.'))
    valid={}
    for model in ('ouro','huginn'):
        for method in ('chunked','paced'):
            cfg=configuration(model,method,8,verify=True)
            cfg.update(name=f'qual8_{model}_{method}_c128_v4',dlocal_chunk=128,
                prime_all_graph_shapes=False,deep_serial_primary_warmup=False)
            valid[model,method]=dispatch(cfg,'C128_N8') and compare(
                f'local_{model}_native_n8_s2026110701_off_state',cfg['name'],8)['passed']
    for model in ('ouro','huginn'):
        reference=f'qual32_{model}_native_v3'
        receipt=ROOT/'raw'/reference/'dispatcher_result.json'
        while not receipt.exists():
            assert time.time()<LONG_START_CUTOFF,'Reference wait reached bounded start cutoff'
            time.sleep(5)
        if not json.loads(receipt.read_text()).get('complete'):
            continue
        for method in ('chunked','paced'):
            if not valid.get((model,method)):continue
            cfg=configuration(model,method,32,verify=True)
            cfg.update(name=f'qual32_{model}_{method}_c128_v4',arrival_offsets=[0.]*32,
                dlocal_chunk=128,prime_all_graph_shapes=False,deep_serial_primary_warmup=False)
            valid[model,method]=dispatch(cfg,'C128_N32') and compare(reference,cfg['name'],32)['passed']
    for model in ('ouro','huginn'):
        for method in ('full','chunked','paced'):
            qualification=ROOT/'raw'/f'qual32_{model}_full_scoped_v3/local_qualification.json' if method=='full' else None
            passed=(qualification.exists() and json.loads(qualification.read_text())['passed']) if method=='full' else valid.get((model,method))
            if not passed:continue
            count=32 if method=='full' else 128
            cfg=configuration(model,method,count)
            cfg.update(name=f'serve_{model}_{method}_c128_fixed25_v4',dlocal_chunk=128,
                dlocal_every=4,dlocal_max_active=1)
            dispatch(cfg,'NAIVE_DIAGNOSTIC' if method=='full' else 'MARKER_FREE_GENERIC_N128')
    write(ROOT/'state/controller_v4.json',dict(stage='COMPLETE',time=time.time()))


if __name__=='__main__':main()
