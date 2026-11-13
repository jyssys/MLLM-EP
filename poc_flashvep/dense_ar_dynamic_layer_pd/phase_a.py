"""Qualify strong baselines before tuning Dynamic layer continuation."""
import argparse,fcntl
from .common import *
from .prepare import base,prepare_data
from .run import execute
from .analyze import metrics,compare


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--initial',action='store_true');args=parser.parse_args()
    prepare_data();results=[]
    ref='qual_native_O200_N8_A_v1'
    jobs=[]
    for suffix in ('A','B'):
        c=base(f'qual_native_O200_N8_{suffix}_v1',count=8,cap=200,dataset='qualification.json')
        c.update(arrival_offsets=[0.]*8,arrival_rate=None,phase='FRESH_NATIVE_EXACT_OUTPUT_CONTROL')
        jobs.append(c)
    c=base('capacity_P_B124_v1',count=8,dataset='qualification.json')
    c.update(module='capacity',capacity_mode='P_ONLY',gpus=[0],jobs=[dict(batch=b,cap=1,repeats=3) for b in (1,2,4)],
        reuse_prefill_token=False,serialize_D_imports=False,cohort_barrier=False)
    jobs.append(c)
    c=base('capacity_D_O200_B14816_v1',count=8,cap=200,dataset='calibration.json')
    c.update(module='capacity',capacity_mode='PREPARED_D',gpus=[1],jobs=[dict(batch=b,cap=200,repeats=2) for b in (1,4,8,16)],
        reuse_prefill_token=False,serialize_D_imports=False,cohort_barrier=False,wall_timeout=1200)
    jobs.append(c)
    for label,budget,chunk,cohort,slots in (
        ('cohort4',32000,8000,True,16),('chunk512',2048,512,False,16),
        ('chunk1024',4096,1024,False,16),('chunk2048',8192,2048,False,32)):
        c=base(f'qual_coloc_{label}_O200_N8_v1',arm='coloc_cohort' if cohort else 'coloc_chunk',
            count=8,cap=200,dataset='qualification.json')
        c.update(arrival_offsets=[0.]*8,arrival_rate=None,token_budget=budget,chunk=chunk,D_slots=slots,
            atomic_prompt=False,warmup_batch_sizes=[] if not cohort else [1,2,4],phase='STRONG_COLOCATED_CORRECTNESS_AND_TUNING')
        jobs.append(c)
    write(ROOT/'state/phase_A_initial_plan.json',dict(jobs=[save(c) for c in jobs]))
    try:
        for c in jobs:
            name=c['name']
            if not execute(name,restore=False):
                write(ROOT/'state/phase_A_initial_progress.json',dict(results=results,failed_job=name));return
            if c.get('module')=='capacity':
                row=dict(name=name,capacity_complete=True)
            else:
                row=dict(name=name,metrics=metrics(name))
                if name!=ref:row['output_gate']=compare(ref,name)
                if name=='qual_native_O200_N8_B_v1' and not row['output_gate']['passed']:
                    results.append(row);write(ROOT/'state/phase_A_initial_progress.json',dict(results=results,stop='NATIVE_REFERENCE_UNSTABLE'));return
            results.append(row);write(ROOT/'state/phase_A_initial_progress.json',dict(results=results))
            print('PHASE_A_INITIAL',name,row.get('output_gate',{}).get('passed','CAPACITY_OR_REFERENCE'),flush=True)
    finally:
        with QUIET_LOCK.open('a+') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX)
            if not resources().processes():resources().restore(list(range(8)))


if __name__=='__main__':main()
