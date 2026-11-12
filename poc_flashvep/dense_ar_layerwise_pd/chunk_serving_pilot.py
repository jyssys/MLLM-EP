"""Benchmark only exact-output qualified colocated chunk configurations."""
import fcntl,json,time
from .common import *
from .prepare import save
from .run import execute
from .analyze import metrics,compare


def main():
    native='serving_qwen8_L8000_O16_N64_s3_native_pilot_v1'
    template=json.loads((ROOT/'configs'/(native+'.json')).read_text());results=[];jobs=[]
    try:
        for chunk in (64,128,256,512,1024,2048):
            while True:
                progress=json.loads((ROOT/'state/common_compile_chunk_progress.json').read_text())['results']
                rows=[x for x in progress if x['chunk']==chunk and x['count']==32]
                if rows:break
                assert time.time()<LONG_START_CUTOFF
                time.sleep(5)
            if not rows[0]['passed']:continue
            c=dict(template,name=f'serving_qwen8_L8000_O16_N64_s3_chunk{chunk}_common_compile_v3',
                arm='DP2',chunk=chunk,token_budget=32000,atomic_prompt=False,layer_mode=None,
                hidden_transport=None,serialize_D_imports=False,cohort_barrier=False,
                phase='QUALIFIED_COLOCATED_CHUNK_SERVING_SELECTION',
                scope='Two full-model colocated replicas. Per-request chunk threshold with common 32000 token budget. Actual aggregate rows are recorded by the Native scheduler. No P-to-D token suffix migration.')
            name=save(c);jobs.append(name)
            write(ROOT/'state/chunk_serving_pilot_plan.json',dict(jobs=[native]+jobs))
            if not execute(name,restore=False):break
            row=dict(name=name,metrics=metrics(name),output_gate=compare('native_qwen8_L8000_O16_N128_s4_cohort_cleanA_v4',name,True))
            results.append(row);write(ROOT/'state/chunk_serving_pilot_progress.json',dict(results=results))
            print('CHUNK_SERVING',chunk,{k:row['metrics'][k] for k in ('TTFT_p95_ms','ITL_p99_ms','TPOT_p95_ms','maximum_ITL_ms')},row['output_gate']['passed'],flush=True)
    finally:
        with (ROOT/'state/quiet_lane.lock').open('a+') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX)
            if not resources().processes():resources().restore(list(range(8)))


if __name__=='__main__':main()
