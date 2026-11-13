"""Test measured batch costs without changing execution or safety gates."""
import fcntl
from .common import *
from .online_pilots import serving_config
from .run import execute
from .analyze import compare,metrics,records


def main():
    initialize();jobs=[];results=[]
    c=serving_config('predictor_COST_BATCH_O32_N32_qualification_v1',cap=32,count=32,
        decode_prediction='BATCH_LOOKUP',decode_prediction_margin_ms=2.,verify_layer_execution=True,
        phase='PAST_ONLY_BATCH_COST_PREDICTOR_N32_EXACTNESS')
    jobs.append(c)
    for cap in (16,32,64):
        jobs.append(serving_config(f'predictor_COST_BATCH_O{cap}_N64_v1',cap=cap,
            decode_prediction='BATCH_LOOKUP',decode_prediction_margin_ms=2.,
            phase='BATCH_COST_PREDICTOR_DISCOVERY_WITH_FIXED_SLO_AND_COHORT_GUARD'))
    try:
        for c in jobs:
            if not execute(save(c),restore=False):return
            gate=compare(f'coarse_O{c["cap"]}_N64_q4.777_native16_s1_v1',c['name'],c['count']==32)
            commits=records(ROOT/'raw'/c['name']/'actual_cut_commits.jsonl')
            distribution={str(n):sum(len(r['requests']) for r in commits if r['offloaded_layers']==n) for n in (0,2,4,8,12)}
            row=dict(name=c['name'],output_gate=gate,metrics=metrics(c['name']),actual_layer_distribution=distribution)
            results.append(row);write(ROOT/'state/predictor_refinement_progress.json',dict(results=results))
            print('PREDICTOR_REFINEMENT',c['name'],gate['passed'],distribution,flush=True)
            if not gate['passed']:return
        write(ROOT/'analysis/predictor_refinement_results.json',dict(results=results,
            scope='Lookup uses frozen measured batch costs plus2ms. Unknown future arrivals remain unknown. The actual destination still waits for the existing Decode cohort to finish.'))
    finally:
        with QUIET_LOCK.open('a+') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX)
            if not resources().processes():resources().restore(list(range(8)))


if __name__=='__main__':main()
