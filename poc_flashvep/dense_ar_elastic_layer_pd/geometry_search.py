"""Separate destination row capacity from the source execution budget."""
import fcntl
from .common import *
from .configuration import base
from .run import execute
from .analyze import compare,metrics,records


def plan():
    jobs=[]
    for cap,destination in ((64,8),(64,16),(200,16),(16,8)):
        common=dict(D_admission_batch=destination,D_token_budget=destination*8000,
            max_reserved_requests=32,phase='DESTINATION_ROW_GEOMETRY_DISCOVERY')
        if cap!=16:
            jobs.append(base(f'geometry_Native_D{destination}_O{cap}_N64_v2',cap=cap,**common))
        cuts=(4,8,12) if cap!=200 else (2,8,12)
        for layers in cuts:
            jobs.append(base(f'geometry_Static{layers}_D{destination}_O{cap}_N64_v2',
                arm='STATIC',cap=cap,static_layers=layers,migration_fraction=1.,
                revised_revision=2,cost_model='revised_cost_model_v2.json',**common))
        for wait in ((0,250) if cap==64 else (250,)):
            jobs.append(base(f'geometry_Revised_D{destination}_W{wait}_O{cap}_N64_v2',
                arm='Revised',cap=cap,revised_revision=2,cost_model='revised_cost_model_v2.json',
                maximum_predicted_D_wait_ms=wait,maximum_queue_groups=8,**common))
    for cfg in jobs:save(cfg)
    write(ROOT/'state/geometry_search_plan.json',dict(jobs=[c['name'] for c in jobs],
        scope='Viewed prompt discovery. Both Static and Dynamic use the same revised legal destination capacity. No held-out request is used.'))
    return jobs


def main():
    initialize();assert (ROOT/'configs/revised_cost_model_v2.json').exists()
    results=[]
    try:
        for cfg in plan():
            name=cfg['name']
            if not execute(name,restore=False):
                write(ROOT/'state/geometry_search_progress.json',dict(results=results,failed=name));return
            output=compare('baseline_Native_O200_N64_q4p8_s21_v1',name,True)
            requests=records(ROOT/'raw'/name/'requests.jsonl')
            work={str(n):sum(r.get('completed_D_prefill_layers',0)==n for r in requests) for n in (0,2,4,8,12)}
            row=dict(name=name,output=output,metrics=metrics(name),completed_layer_distribution=work)
            results.append(row);write(ROOT/'state/geometry_search_progress.json',dict(results=results))
            print('GEOMETRY',name,output['passed'],work,
                *[round(row['metrics'][k],2) for k in ('TTFT_p95_ms','ITL_p99_ms','TPOT_p95_ms','output_tokens_s')],flush=True)
            if not output['passed']:return
        write(ROOT/'analysis/geometry_search_results.json',dict(results=results,
            evidence='MEASURED',scope='One-start selection data. This is not independent confirmation.'))
    finally:
        with QUIET_LOCK.open('a+') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX)
            if not resources().processes():resources().restore(list(range(8)))


if __name__=='__main__':main()
