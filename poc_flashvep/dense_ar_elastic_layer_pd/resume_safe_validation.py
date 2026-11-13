"""Replace unexecuted risky ingress variants after their failure is measured."""
import fcntl
from .common import *
from .final_validation import jobs
from .run import execute
from .analyze import compare,metrics,records


def main():
    progress=ROOT/'state/final_validation_progress.json'
    state=json.loads(progress.read_text());rows=state['results'];present={r['name'] for r in rows};not_run=[]
    original=jobs();planned={c['name'] for c in original};queue=[]
    for cfg in original:
        if cfg['name'] in present:continue
        receipt=ROOT/'raw'/cfg['name']/'dispatcher_result.json'
        completed=receipt.exists() and json.loads(receipt.read_text()).get('complete')
        unsafe=cfg['cap']>=64 and cfg.get('native_continuous_D_ingress')
        if unsafe and not completed:
            if cfg['evaluation_arm']=='Native':
                not_run.append(dict(name=cfg['name'],status='NOT_RUN',reason='KNOWN_UNSAFE_NATIVE_INGRESS_ALREADY_MEASURED'));continue
            if cfg['evaluation_arm'] in ('STATIC','Revised'):
                name=cfg['name']
                if cfg['evaluation_arm']=='STATIC' and cfg.get('migration_fraction',1.)<1.:
                    equivalent=name.replace('_f','_cohort_f',1)
                elif cfg['evaluation_arm']=='Revised' and cfg['maximum_predicted_D_wait_ms']==250:
                    equivalent=f'validation_Revised_cohort_IO_fixed_O{cfg["cap"]}_N64_v3'
                else:equivalent=name+'_isolated'
                not_run.append(dict(name=name,status='NOT_RUN',reason='KNOWN_UNSAFE_NATIVE_INGRESS',replacement=equivalent))
                if equivalent in planned:continue
                cfg=dict(cfg,name=equivalent,native_continuous_D_ingress=False,
                    safety_replacement_of=name,phase='DECODE_ISOLATED_REPLACEMENT_AFTER_INGRESS_FAILURE')
                save(cfg)
        queue.append(cfg)
    try:
        for cfg in queue:
            name=cfg['name']
            if name in present:continue
            if not execute(name,restore=False):
                write(progress,dict(results=rows,failed=name,not_run=not_run));return
            output=compare('baseline_Native_O200_N64_q4p8_s21_v1',name,True)
            requests=records(ROOT/'raw'/name/'requests.jsonl')
            row=dict(name=name,output=output,metrics=metrics(name),
                migrated_requests=sum(r.get('completed_D_prefill_layers',0)>0 for r in requests),
                migrated_layers=sum(r.get('completed_D_prefill_layers',0) for r in requests),
                layer_distribution={str(k):sum(r.get('completed_D_prefill_layers',0)==k for r in requests) for k in (0,2,4,8,12)})
            rows.append(row);present.add(name);write(progress,dict(results=rows,not_run=not_run))
            print('SAFE_VALIDATION',name,output['passed'],row['layer_distribution'],
                *[round(row['metrics'][k],2) for k in ('TTFT_p95_ms','ITL_p99_ms','TPOT_p95_ms')],flush=True)
            if not output['passed']:return
        write(ROOT/'analysis/final_viewed_validation.json',dict(results=rows,not_run=not_run,
            scope='Preserve completed unsafe diagnoses. Replace only unexecuted unsafe ingress variants. SLOs and earlier configurations remain unchanged. All selections still precede fresh evaluation.'))
    finally:
        with QUIET_LOCK.open('a+') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX)
            if not resources().processes():resources().restore(list(range(8)))


if __name__=='__main__':main()
