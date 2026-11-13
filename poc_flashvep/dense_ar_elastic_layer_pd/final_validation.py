"""Select settings on viewed prompts after generic correctness and IO fixes."""
import fcntl
from .common import *
from .configuration import base
from .run import execute
from .analyze import compare,metrics,records


def jobs():
    model=json.loads((ROOT/'configs/revised_cost_model_v3.json').read_text());result=[]
    for cap in (16,64,200):
        shared=dict(defer_completion_log=True,D_admission_batch=16,D_token_budget=128000,
            max_reserved_requests=32,native_continuous_D_ingress=True,
            exclude_negotiation_from_D_occupancy=True,
            phase='FINAL_VIEWED_VALIDATION_BEFORE_FRESH_SELECTION')
        result.append(base(f'validation_Native_O{cap}_N64_v1',cap=cap,**shared))
        cohort=dict(shared,native_continuous_D_ingress=False)
        result.append(base(f'validation_Native_cohort_O{cap}_N64_v1',cap=cap,**cohort))
        for budget in ((512,) if cap!=200 else (384,512)):
            result.append(base(f'validation_Colocated_B{budget}_O{cap}_N64_v1',arm='Colocated',cap=cap,
                token_budget=budget,chunk=budget,defer_completion_log=True,
                phase='FINAL_STRONG_CONTINUOUS_VALIDATION'))
        for sequences in (8,32):
            result.append(base(f'validation_Colocated_B512_S{sequences}_O{cap}_N64_v1',arm='Colocated',cap=cap,
                token_budget=512,chunk=512,D_slots=sequences,defer_completion_log=True,
                phase='STRONG_CONTINUOUS_SEQUENCE_LIMIT_VALIDATION'))
        static_options=[(8,1.),(12,1.),(12,.3125),(12,.5)] if cap==16 else [(4,1.),(8,1.),(12,1.),(12,.3125),(12,.5)] if cap==64 else [(8,1.),(12,.3125)]
        for layers,fraction in static_options:
            result.append(base(f'validation_Static{layers}_f{fraction:g}_O{cap}_N64_v1',arm='STATIC',cap=cap,
                static_layers=layers,migration_fraction=fraction,static_fraction_axis='REQUEST_ID_BLOCKS',
                cost_model='revised_cost_model_v3.json',revised_revision=2,**shared))
            if cap in (16,64) and fraction<1.:
                result.append(base(f'validation_Static{layers}_cohort_f{fraction:g}_O{cap}_N64_v1',arm='STATIC',cap=cap,
                    static_layers=layers,migration_fraction=fraction,static_fraction_axis='REQUEST_ID_BLOCKS',
                    cost_model='revised_cost_model_v3.json',revised_revision=2,**cohort))
        for wait in (0,250):
            extra=dict(safety_margin_by_layers_ms=model['Output16_safety_margin_by_layers_ms']) if cap==16 else {}
            result.append(base(f'validation_Revised_W{wait}_O{cap}_N64_v3',arm='Revised',cap=cap,
                revised_revision=2,cost_model='revised_cost_model_v3.json',maximum_predicted_D_wait_ms=wait,
                maximum_queue_groups=8,**extra,**shared))
        extra=dict(safety_margin_by_layers_ms=model['Output16_safety_margin_by_layers_ms']) if cap==16 else {}
        result.append(base(f'validation_Revised_cohort_IO_fixed_O{cap}_N64_v3',arm='Revised',cap=cap,
                revised_revision=2,cost_model='revised_cost_model_v3.json',maximum_predicted_D_wait_ms=250,
                maximum_queue_groups=8,**extra,**cohort))
    for cfg in result:save(cfg)
    write(ROOT/'state/final_validation_plan.json',dict(jobs=[c['name'] for c in result],
        scope='Viewed discovery prompts only. All final methods share the generic deferred-log fix. No fresh confirmation information enters selection.'))
    return result


def main():
    assert json.loads((ROOT/'analysis/native_continuous_qualification_and_discovery.json').read_text())['passed']
    assert json.loads((ROOT/'analysis/continuous_native_mixed_N32_qualification.json').read_text())['passed']
    assert json.loads((ROOT/'analysis/receiver_full_state_qualification_v3.json').read_text())['passed']
    rows=[]
    try:
        for cfg in jobs():
            name=cfg['name']
            if not execute(name,restore=False):write(ROOT/'state/final_validation_progress.json',dict(results=rows,failed=name));return
            output=compare('baseline_Native_O200_N64_q4p8_s21_v1',name,True)
            req=records(ROOT/'raw'/name/'requests.jsonl')
            row=dict(name=name,output=output,metrics=metrics(name),
                migrated_requests=sum(r.get('completed_D_prefill_layers',0)>0 for r in req),
                migrated_layers=sum(r.get('completed_D_prefill_layers',0) for r in req),
                layer_distribution={str(k):sum(r.get('completed_D_prefill_layers',0)==k for r in req) for k in (0,2,4,8,12)})
            rows.append(row);write(ROOT/'state/final_validation_progress.json',dict(results=rows))
            print('FINAL_VALIDATION',name,output['passed'],row['layer_distribution'],
                *[round(row['metrics'][k],2) for k in ('TTFT_p95_ms','ITL_p99_ms','TPOT_p95_ms','output_tokens_s')],flush=True)
            if not output['passed']:return
        write(ROOT/'analysis/final_viewed_validation.json',dict(results=rows,
            scope='Selection data only. No independent-start performance or non-inferiority conclusion.'))
    finally:
        with QUIET_LOCK.open('a+') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX)
            if not resources().processes():resources().restore(list(range(8)))


if __name__=='__main__':main()
