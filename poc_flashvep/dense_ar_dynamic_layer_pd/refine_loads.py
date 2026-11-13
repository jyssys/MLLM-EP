"""Refine load using the strongest exact Colocated pilot for each output."""
import fcntl
from .common import *
from .prepare import base
from .run import execute
from .analyze import metrics,compare


def main():
    initialize();deadline=time.monotonic()+7200
    while not (ROOT/'analysis/strong_baseline_results.json').exists():
        assert time.monotonic()<deadline,'Strong baseline tuning did not finish'
        time.sleep(5)
    rows=json.loads((ROOT/'analysis/strong_baseline_results.json').read_text())['results']
    rows+=json.loads((ROOT/'analysis/layout_control_results.json').read_text())['results']
    chosen={}
    for cap in (16,32,64,128,200):
        candidates=[r for r in rows if r['output_gate']['passed'] and r['metrics']['arm']=='DP2' and r['metrics']['output_cap']==cap]
        assert candidates
        best=min(candidates,key=lambda r:r['metrics']['TTFT_p95_ms'])
        chosen[cap]=json.loads((ROOT/'configs'/(best['name']+'.json')).read_text())
    capacity=json.loads((ROOT/'analysis/initial_capacity_selection.json').read_text())['P_capacity_requests_s']
    rates=[capacity*.7,capacity*1.7]
    write(ROOT/'state/load_refinement_selection.json',dict(selected={str(k):v['name'] for k,v in chosen.items()},
        rates=rates,rule='Use the exact-output Colocated configuration with the lowest discovery TTFT p95. Keep all tuning results. Probe0.7x and1.7x measured P capacity around the existing1.35x pilot.'))
    results=[]
    try:
        for cap in (16,32,64,128,200):
            for rate in rates:
                ref=None
                for arm in ('native','coloc'):
                    c=base(f'refine_O{cap}_N64_q{rate:.3f}_{arm}_v1',cap=cap,rate=rate)
                    if arm=='coloc':
                        template=chosen[cap]
                        for key in ('arm','colocated_cohorts','D_admission_batch','chunk','token_budget','D_slots',
                            'warmup_batch_sizes','reuse_prefill_token','serialize_D_imports','cohort_barrier'):c[key]=template[key]
                    else:c['D_admission_batch']=16
                    c.update(kv_layout='HND',hidden_transport=None,phase='CAPACITY_GUIDED_QPS_REFINEMENT')
                    if not execute(save(c),restore=False):return
                    row=dict(name=c['name'],metrics=metrics(c['name']))
                    if arm=='native':ref=c['name']
                    else:row['output_gate']=compare(ref,c['name'])
                    results.append(row);write(ROOT/'state/load_refinement_progress.json',dict(results=results))
                    print('QPS_REFINEMENT',c['name'],row.get('output_gate',{}).get('passed'),row['metrics']['TTFT_p95_ms'],row['metrics']['ITL_p99_ms'],flush=True)
        write(ROOT/'analysis/load_refinement_results.json',dict(results=results))
    finally:
        with QUIET_LOCK.open('a+') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX)
            if not resources().processes():resources().restore(list(range(8)))


if __name__=='__main__':main()
