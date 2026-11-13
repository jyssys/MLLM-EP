"""Tune exact Colocated execution without weakening its admission policy."""
import fcntl
from .common import *
from .prepare import base
from .run import execute
from .analyze import metrics,compare


def main():
    initialize();started=time.monotonic()
    while not (ROOT/'state/mixed_protocol_qualified.json').exists():
        assert time.monotonic()-started<2400,'Protocol qualification did not pass'
        time.sleep(5)
    jobs=[]
    c=base('capacity_D_graph_HND_lean_all_caps_v3',count=16,cap=200,dataset='calibration.json')
    c.update(module='capacity',capacity_mode='PREPARED_D',gpus=[1],reuse_prefill_token=False,
        serialize_D_imports=False,cohort_barrier=False,decode_only_graph_roles=['capacity'],wall_timeout=1500,
        hidden_transport=None,kv_layout='HND',
        jobs=[dict(batch=b,cap=cap,repeats=2) for cap in (16,32,64,128,200) for b in ((1,4,8,16) if cap in (16,200) else (4,16))],
        phase='MATCHED_HND_DECODE_GRAPH_LEAN_CAPACITY')
    jobs.append(c)
    for cap in (16,32,64,128,200):
        for mode,maximum,budget in (('cohort',8,64000),('cohort',16,128000),('stream',16,32000)):
            c=base(f'strong_O{cap}_N64_{mode}{maximum}_v1',arm='coloc_cohort' if mode=='cohort' else 'coloc',cap=cap,rate=4.777100014491616)
            c.update(kv_layout='HND',hidden_transport=None,token_budget=budget,D_slots=16,
                D_admission_batch=maximum,warmup_batch_sizes=[1,2,4,8] if maximum>=8 else [1,2,4],
                phase='STRONG_COLOCATED_CAPACITY_AND_STREAMING_QUALIFICATION')
            jobs.append(c)
    write(ROOT/'state/strong_baseline_plan.json',dict(jobs=[save(c) for c in jobs]))
    results=[]
    try:
        for c in jobs:
            if not execute(c['name'],restore=False):return
            if c.get('module')=='capacity':continue
            ref=next(p.stem for p in (ROOT/'configs').glob(f'coarse_O{c["cap"]}_N64_*native16_s1_v1.json'))
            row=dict(name=c['name'],output_gate=compare(ref,c['name']),metrics=metrics(c['name']))
            results.append(row);write(ROOT/'state/strong_baseline_progress.json',dict(results=results))
            print('STRONG_BASELINE',c['name'],row['output_gate']['passed'],row['metrics']['TTFT_p95_ms'],row['metrics']['ITL_p99_ms'],flush=True)
        write(ROOT/'analysis/strong_baseline_results.json',dict(results=results))
    finally:
        with QUIET_LOCK.open('a+') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX)
            if not resources().processes():resources().restore(list(range(8)))


if __name__=='__main__':main()
