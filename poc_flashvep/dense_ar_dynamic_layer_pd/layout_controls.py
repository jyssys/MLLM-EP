"""Remove baseline layout and unused-agent differences before policy claims."""
import fcntl
from .common import *
from .run import execute
from .analyze import metrics,compare


def main():
    initialize();started=time.monotonic()
    while not (ROOT/'analysis/coarse_crossover_pairs.json').exists():
        assert time.monotonic()-started<3600,'Coarse output sweep did not finish'
        time.sleep(5)
    results=[];jobs=[]
    for variant,layout,hidden in (('HND_only','HND','NIXL_GPU'),('lean_only','NHD',None),('HND_lean','HND',None)):
        path=next((ROOT/'configs').glob('coarse_O32_N64_*cohort4_s1_v1.json'))
        c=json.loads(path.read_text());c.update(name=f'layout_O32_N64_cohort4_{variant}_v1',
            kv_layout=layout,hidden_transport=hidden,phase='ONE_VARIABLE_BASELINE_LAYOUT_AGENT_AUDIT')
        jobs.append(c)
    for cap in (16,32,64):
        path=next((ROOT/'configs').glob(f'coarse_O{cap}_N64_*native16_s1_v1.json'))
        c=json.loads(path.read_text());c.update(name=f'lean_native_O{cap}_N64_v1',kv_layout='HND',hidden_transport=None,
            phase='STRONG_NATIVE_WITHOUT_UNUSED_HIDDEN_AGENT')
        jobs.append(c)
    for chunk in (512,1024,2048):
        path=next((ROOT/'configs').glob(f'coarse_O16_N64_*chunk{chunk}_s1_v1.json'))
        c=json.loads(path.read_text());c.update(name=f'HND_lean_chunk{chunk}_O16_N64_v1',kv_layout='HND',hidden_transport=None,
            phase='MATCHED_HND_STRONG_CHUNK_QUALIFICATION')
        jobs.append(c)
    write(ROOT/'state/layout_control_plan.json',dict(jobs=[save(c) for c in jobs],
        rationale='Native NIXL forces HND. Standalone default and unused optional agents are potential baseline confounders. Keep frozen SLOs unchanged.'))
    try:
        for c in jobs:
            name=c['name']
            if not execute(name,restore=False):return
            cap=c['cap'];ref=next(p.stem for p in (ROOT/'configs').glob(f'coarse_O{cap}_N64_*native16_s1_v1.json'))
            gate=compare(ref,name)
            row=dict(name=name,output_gate=gate,metrics=metrics(name));results.append(row)
            write(ROOT/'state/layout_control_progress.json',dict(results=results))
            print('LAYOUT_CONTROL',name,gate['passed'],row['metrics']['TTFT_p95_ms'],row['metrics']['ITL_p99_ms'],flush=True)
        write(ROOT/'analysis/layout_control_results.json',dict(results=results,SLO_unchanged=True,
            scope='Generic baseline/runtime tuning. Do not attribute its effects to Layer offloading.'))
    finally:
        with QUIET_LOCK.open('a+') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX)
            if not resources().processes():resources().restore(list(range(8)))


if __name__=='__main__':main()
