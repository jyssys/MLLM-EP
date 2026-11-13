"""Check smaller chunk budgets and a full-model compiled baseline."""
import fcntl
from .common import *
from .prepare import base
from .run import execute
from .analyze import metrics,compare


def main():
    initialize();jobs=[]
    for chunk in (128,256):
        c=base(f'strong_small_global_chunk{chunk}_O16_N64_v1',arm='coloc',cap=16,rate=4.777100014491616)
        c.update(chunk=chunk,token_budget=chunk,atomic_prompt=False,hidden_transport=None,kv_layout='HND',
            phase='SMALL_GLOBAL_BUDGET_STRONG_CHUNK_CONTROL')
        jobs.append(c)
    for arm in ('native','coloc_cohort'):
        c=base(f'qual_compiled_full_{arm}_O16_N8_v1',arm=arm,count=8,cap=16,dataset='qualification.json')
        c.update(arrival_offsets=[0.]*8,arrival_rate=None,kv_layout='HND',hidden_transport=None,
            ar_compile_mode=3,ar_execution_family='COMPILED_FULL_MODEL_CONTROL',isolate_layer_compile_cache=True,
            wall_timeout=1800,startup_timeout=800,D_admission_batch=4,
            phase='FULL_MODEL_COMPILE_CORRECTNESS_BEFORE_STRONG_BASELINE_PERFORMANCE')
        jobs.append(c)
    results=[]
    try:
        for c in jobs:
            if not execute(save(c),restore=False):
                results.append(dict(name=c['name'],execution='FAILED',output_gate=dict(passed=False)));continue
            ref='qual_native_O200_N8_A_v1' if c['count']==8 else 'coarse_O16_N64_q4.777_native16_s1_v1'
            gate=compare(ref,c['name'],c['count']==8)
            row=dict(name=c['name'],output_gate=gate,metrics=metrics(c['name']))
            results.append(row);write(ROOT/'state/strong_kernel_control_progress.json',dict(results=results))
            print('STRONG_KERNEL_CONTROL',c['name'],gate['passed'],flush=True)
        write(ROOT/'analysis/strong_kernel_control_results.json',dict(results=results,
            scope='A failed output configuration is diagnostic only. A qualified compiled full-model baseline must enter confirmation before any strongest-baseline claim.'))
    finally:
        with QUIET_LOCK.open('a+') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX)
            if not resources().processes():resources().restore(list(range(8)))


if __name__=='__main__':main()
