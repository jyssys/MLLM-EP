"""Keep full static fast paths and distributed fraction controls in validation."""
import fcntl
from .common import *
from .prepare import base
from .qualify import static_config
from .online_pilots import serving_config
from .run import execute
from .analyze import metrics,compare,records


def main():
    initialize();deadline=time.monotonic()+10800
    while not (ROOT/'analysis/online_pilot_results.json').exists():
        assert time.monotonic()<deadline,'Online policy pilots did not finish'
        time.sleep(5)
    jobs=[]
    for cap in (16,32,64):
        for layers in (2,4,8,12):
            c=static_config(layers,count=64,diagnostic=False)
            b=base('unused',cap=cap,rate=4.777100014491616)
            c.update(name=f'validation_fast_Static{layers}_O{cap}_N64_v1',dataset='discovery.json',cap=cap,
                request_caps=[cap]*64,arrival_offsets=b['arrival_offsets'],arrival_rate=b['arrival_rate'],
                arrival_seed=b['arrival_seed'],kv_layout='HND',execution_events=False,D_admission_batch=4,
                hidden_destination_slots=16,phase='STRONG_FULL_STATIC_FAST_PATH_VALIDATION')
            jobs.append(c)
        if cap in (16,32,64):
            for fraction in (.25,.5):
                for layers in (2,4,8,12):
                    c=serving_config(f'validation_fraction_Static{layers}_f{int(100*fraction)}_O{cap}_N64_v1',
                        cap=cap,policy='STATIC',static_layers=layers,migration_fraction=fraction,
                        phase='DISTRIBUTED_STATIC_FRACTION_VALIDATION')
                    jobs.append(c)
        if cap==16:
            jobs.append(serving_config('validation_matched_Static12_f31p25_O16_N64_v1',cap=16,
                policy='STATIC',static_layers=12,migration_fraction=.3125,
                phase='DISCOVERY_FRACTION_MATCHED_TO_COST_POLICY_NOT_HELDOUT_TUNING'))
    deadline=time.monotonic()+2400
    while not (ROOT/'state/D8_mixed_protocol_qualified.json').exists():
        assert time.monotonic()<deadline,'Wider destination mixed qualification did not finish'
        time.sleep(5)
    if json.loads((ROOT/'state/D8_mixed_protocol_qualified.json').read_text())['passed']:
        for cap in (16,32,64):
            for layers in (2,4,8,12):
                c=static_config(layers,count=64,diagnostic=False)
                b=base('unused',cap=cap,rate=4.777100014491616)
                c.update(name=f'validation_D8_fast_Static{layers}_O{cap}_N64_v1',dataset='discovery.json',cap=cap,
                    request_caps=[cap]*64,arrival_offsets=b['arrival_offsets'],arrival_rate=b['arrival_rate'],
                    arrival_seed=b['arrival_seed'],kv_layout='HND',execution_events=False,D_admission_batch=8,
                    D_token_budget=64000,hidden_destination_slots=16,
                    phase='WIDER_QUALIFIED_STRONG_FULL_STATIC_VALIDATION')
                jobs.append(c)
            for fraction in (.25,.5):
                jobs.append(serving_config(f'validation_D8_fraction_Static12_f{int(100*fraction)}_O{cap}_N64_v1',
                    cap=cap,policy='STATIC',static_layers=12,migration_fraction=fraction,
                    D_token_budget=64000,phase='WIDER_DESTINATION_STATIC_FRACTION_VALIDATION'))
    write(ROOT/'state/static_validation_plan.json',dict(jobs=[save(c) for c in jobs]))
    results=[]
    try:
        for c in jobs:
            if not execute(c['name'],restore=False):return
            ref=next(p.stem for p in (ROOT/'configs').glob(f'coarse_O{c["cap"]}_N64_*native16_s1_v1.json'))
            gate=compare(ref,c['name']);rows=records(ROOT/'raw'/c['name']/'requests.jsonl')
            migrated=sum(r.get('completed_D_prefill_layers',36-c['layer_cut']) for r in rows)
            row=dict(name=c['name'],output_gate=gate,metrics=metrics(c['name']),completed_migrated_layers=migrated)
            results.append(row);write(ROOT/'state/static_validation_progress.json',dict(results=results))
            print('STATIC_VALIDATION',c['name'],gate['passed'],row['metrics']['TTFT_p95_ms'],row['metrics']['ITL_p99_ms'],flush=True)
            if not gate['passed']:return
        write(ROOT/'analysis/static_validation_results.json',dict(results=results))
    finally:
        with QUIET_LOCK.open('a+') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX)
            if not resources().processes():resources().restore(list(range(8)))


if __name__=='__main__':main()
