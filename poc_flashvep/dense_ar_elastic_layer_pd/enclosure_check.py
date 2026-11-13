"""Compare stream-event diagnostics with adjacent marker-free controls."""
from .common import *
from .configuration import base
from .run import execute
from .analyze import compare,metrics


def execute_checks(selection,reference):
    path=ROOT/'state/enclosure_progress.json'
    state=json.loads(path.read_text()) if path.exists() else dict(results=[],not_run=[])
    present={r['name'] for r in state['results']}
    for arm in ('Native','Colocated','Static','Revised'):
        settings=dict(selection['settings_by_cap']['16'][arm])
        runtime_arm=settings.pop('_base_arm','STATIC' if arm=='Static' else arm)
        for variant in (('A','events','C') if arm in ('Native','Revised') else ('events',)):
            name=f'enclosure_{arm}_O16_N64_{variant}_v2'
            cfg=base(name,arm=runtime_arm,count=64,cap=16,rate=4.8,
                seed=selection['arrival_seeds'][0],dataset='fresh_confirmation.json',
                phase='STREAM_EVENT_INTERFERENCE_CONTROL_NOT_PRIMARY',**settings)
            cfg.update(execution_events=variant=='events',boundary_events=variant=='events' and arm!='Colocated')
            save(cfg)
            if name in present:continue
            if time.time()>=selection['experiment_deadline_epoch']:
                state['not_run'].append(dict(name=name,status='NOT_RUN',reason='CAMPAIGN_EXPERIMENT_DEADLINE'));continue
            if not execute(name,restore=False):state['failed']=name;write(path,state);return False
            output=compare(reference,name,True)
            state['results'].append(dict(name=name,arm=arm,variant=variant,output=output,metrics=metrics(name)))
            write(path,state);print('ENCLOSURE',name,output['passed'],flush=True)
            if not output['passed']:return False
    groups=[]
    for arm in ('Native','Colocated','Static','Revised'):
        values={r['variant']:r for r in state['results'] if r['arm']==arm}
        if set(values)!={'A','events','C'}:continue
        a,b,c=(values[k]['metrics'] for k in ('A','events','C'))
        keys=('TTFT_p95_ms','E2E_p95_ms','ITL_p99_ms','TPOT_p95_ms','maximum_ITL_ms')
        ratios={k:b[k]/((a[k]+c[k])/2) for k in keys}
        aa={k:c[k]/a[k] for k in keys}
        groups.append(dict(arm=arm,capture_to_mean_controls=ratios,A_A_ratios=aa,
            passed=all(.95<=v<=1.05 for group in (ratios,aa) for v in group.values())))
    write(ROOT/'analysis/enclosure_interference_controls.json',dict(results=groups,
        scope='Even a passed stream-event control does not establish exclusive kernel union or physical idle. Each timeline uses its own recorded start.'))
    return True
