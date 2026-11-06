"""Use the same native full-context decode warm-up policy in all clean arms."""
import time
from .prepare import *
from .run import execute


def main():
    initialize()
    import sys
    full_only=sys.argv[1:]==['full']
    write(ROOT/'state/common_serving_warmup.json',dict(time=time.time(),
        primary='V5 native full-context D decode geometry warm-up enabled in every arm. Common graph budget 8.',
        diagnostic='V4 disabled this extra warm-up in D-local arms. Keep V4 results as tuning diagnostics, not the primary fairness comparison.',
        exactness='C128 N8/N32 math qualification remains mandatory. Clean N128 output parity is checked separately. Warm-up requests are excluded.'))
    for model in ('ouro','huginn'):
        for method in (('full',) if full_only else ('chunked','paced')):
            gate=ROOT/'raw'/(f'qual32_{model}_full_scoped_v3' if method=='full' else f'qual32_{model}_{method}_c128_v4')/'local_qualification.json'
            while not gate.exists():
                state=ROOT/'state/controller_v4.json'
                if state.exists() and json.loads(state.read_text()).get('stage')=='COMPLETE':break
                if time.time()>=LONG_START_CUTOFF:return
                time.sleep(10)
            if not gate.exists() or not json.loads(gate.read_text())['passed']:continue
            cfg=configuration(model,method,32 if full_only else 128)
            cfg.update(name=f'serve_{model}_{method}_c128_fixed25_v5',dlocal_chunk=128,dlocal_every=4,
                dlocal_max_active=1,deep_complete_D_geometry_warmup=True)
            save(cfg);execute(cfg['name'])
    write(ROOT/'state'/('fair_full_complete.json' if full_only else 'fair_generic_complete.json'),dict(time=time.time()))


if __name__=='__main__':main()
