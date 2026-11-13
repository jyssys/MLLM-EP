"""Run frozen choices on unseen prompts with paired independent starts."""
import fcntl
from .common import *
from .configuration import base
from .run import execute
from .analyze import compare,metrics,records


def runtime_files():
    names=('common','native','dynamic_native','extension','dynamic_extension','worker','credits','policy',
        'phase_stable_norm','ack_retention','output_priming','hidden_transport','token_protocol','async_ingress',
        'watch','boundaries','state_witness','nsys_window','secondary_controls','execution_cut','imported_native_state')
    return [SOURCE/(n+'.py') for n in names]


def freeze(selection):
    path=ROOT/'configs/heldout_selection.json'
    assert not path.exists(),'Do not revise a held-out selection'
    selection.update(time=time.time(),prompt_set='fresh_confirmation.json',
        prompt_SHA256=digest(ROOT/'data/fresh_confirmation.json'),
        arrival_seeds=[2026111381,2026111382,2026111383],
        grid={'16':[4.8,2.5,6.4],'64':[4.8,2.5,6.4],'200':[4.8,1.5,2.5]},
        independent_starts_per_cell=3,request_count_per_start=128,
        experiment_deadline_epoch=json.loads((ROOT/'state/progress.json').read_text())['start_epoch']+10.25*3600,
        scope='Selection uses viewed discovery data only. Each method receives the same unseen prompts and arrival trace. Frozen absolute SLOs remain unchanged.')
    write(path,selection)
    paths=runtime_files()+[path,ROOT/'data/fresh_confirmation.json',ROOT/'configs/frozen_SLO_regimes.json']
    paths+=[ROOT/'configs'/n for n in ('revised_cost_model_v1.json','revised_cost_model_v2.json','revised_cost_model_v3.json','calibrated_cost_model_previous.json')]
    write(ROOT/'state/heldout_runtime_freeze.json',dict(time=time.time(),hashes={str(p.relative_to(REPO)):digest(p) for p in paths},
        scope='Freeze execution code, predictors, SLO, prompt pool, and selected settings before any fresh evaluation. Analysis and report code can change.'))


def plan(selection):
    jobs=[]
    reference=base('heldout_reference_Native_O200_N128_v1',count=128,cap=200,rate=2.5,
        seed=2026111380,dataset='fresh_confirmation.json',**selection['native_settings'])
    jobs.append(('Reference',reference))
    cells=[(int(cap),rate) for cap,rates in selection['grid'].items() for rate in rates]
    cells.sort(key=lambda v:(0 if v==(64,4.8) else 1 if v==(16,4.8) else 2 if v[0]==16 else 3 if v[0]==64 else 4,selection['grid'][str(v[0])].index(v[1])))
    for cap,rate in cells:
            for replicate,seed in enumerate(selection['arrival_seeds']):
                arms=['Native','Colocated','Static','Revised']
                offset=replicate%len(arms);arms=arms[offset:]+arms[:offset]
                if replicate%2:arms.reverse()
                for arm in arms:
                    settings=dict(selection['settings_by_cap'][str(cap)][arm])
                    runtime_arm=settings.pop('_base_arm','STATIC' if arm=='Static' else arm)
                    cfg=base(f'heldout_{arm}_O{cap}_N128_q{rate:g}_s{seed}_v1',
                        arm=runtime_arm,count=128,cap=cap,rate=rate,seed=seed,
                        dataset='fresh_confirmation.json',phase='FROZEN_FRESH_QPS_CONFIRMATION',
                        **settings)
                    cfg['selection_SHA256']=digest(ROOT/'configs/heldout_selection.json')
                    jobs.append((arm,cfg))
    for _,cfg in jobs:save(cfg)
    write(ROOT/'state/confirmation_plan.json',dict(jobs=[dict(arm=arm,name=cfg['name']) for arm,cfg in jobs],
        scope='Balanced paired order. Each row is an independent server start. N128 throughput includes drain.'))
    return jobs


def main():
    selection=json.loads((ROOT/'configs/heldout_selection.json').read_text());rows=[];not_run=[]
    jobs=plan(selection);reference=jobs[0][1]['name']
    try:
        for arm,cfg in jobs:
            name=cfg['name']
            if time.time()>=selection['experiment_deadline_epoch']:
                not_run.append(dict(name=name,status='NOT_RUN',reason='CAMPAIGN_EXPERIMENT_DEADLINE'));continue
            if not execute(name,restore=False):
                write(ROOT/'state/confirmation_progress.json',dict(results=rows,failed=name,not_run=not_run));return
            gate=compare(reference,name,True)
            requests=records(ROOT/'raw'/name/'requests.jsonl')
            work={str(n):sum(r.get('completed_D_prefill_layers',0)==n for r in requests) for n in (0,2,4,8,12)}
            row=dict(name=name,arm=arm,cap=cfg['cap'],rate=cfg['arrival_rate'],seed=cfg['arrival_seed'],
                output=gate,metrics=metrics(name),completed_layer_distribution=work)
            rows.append(row);write(ROOT/'state/confirmation_progress.json',dict(results=rows,not_run=not_run))
            print('CONFIRMATION',name,gate['passed'],work,
                *[round(row['metrics'][k],2) for k in ('TTFT_p95_ms','ITL_p99_ms','TPOT_p95_ms','output_tokens_s')],flush=True)
            if not gate['passed']:return
            if arm=='Revised' and cfg['cap']==16 and cfg['arrival_rate']==4.8:
                from .secondary_controls import execute_controls
                if not execute_controls(name,selection,reference):return
                if cfg['arrival_seed']==selection['arrival_seeds'][-1]:
                    from .enclosure_check import execute_checks
                    if not execute_checks(selection,reference):return
        write(ROOT/'analysis/confirmation_results.json',dict(results=rows,not_run=not_run,
            evidence='MEASURED',scope='Unseen prompts. Independent paired server starts. Separate primary curves from the longer-output reference.'))
    finally:
        with QUIET_LOCK.open('a+') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX)
            if not resources().processes():resources().restore(list(range(8)))


if __name__=='__main__':main()
