"""Try actual per-request chunks with the common large compile range."""
import fcntl,json
from .common import *
from .prepare import base,save
from .run import execute
from .analyze import compare


def main():
    results=[];qualified=set();jobs=[]
    for count in (8,32):
        for chunk in (64,128,256,512,1024,2048):
            c=base(f'colocated_qwen8_L8000_O16_threshold{chunk}_budget32000_N{count}_qual_v2',count=count,cap=16)
            c.update(arm='DP2',chunk=chunk,token_budget=32000,atomic_prompt=False,
                fixed_KV_bytes=42*1024**3,batch_commands=True,
                warmup_batch_sizes=[1,2,3,4],warmup_output_cap=16,
                phase='COMMON_COMPILE_RANGE_CHUNK_QUALIFICATION',timeout=360,
                scope='Per-request threshold is enforced by the Native scheduler. Global budget stays 32000, as in PD. Actual aggregate Prefill rows can exceed one threshold and must be reported. This is not a fixed global quantum.')
            jobs.append(save(c))
    write(ROOT/'state/common_compile_chunk_plan.json',dict(jobs=jobs))
    try:
        for n in jobs:
            c=json.loads((ROOT/'configs'/(n+'.json')).read_text());chunk=c['chunk']
            if c['count']==32 and chunk not in qualified:continue
            if not execute(n,restore=False):break
            gate=compare('native_qwen8_L8000_O32_reference32_s0',n,True)
            results.append(dict(name=n,chunk=chunk,count=c['count'],passed=gate['passed'],output_gate=gate))
            write(ROOT/'state/common_compile_chunk_progress.json',dict(results=results))
            print('COMMON_COMPILE_CHUNK',chunk,c['count'],gate['passed'],flush=True)
            if c['count']==8 and gate['passed']:qualified.add(chunk)
    finally:
        with (ROOT/'state/quiet_lane.lock').open('a+') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX)
            if not resources().processes():resources().restore(list(range(8)))


if __name__=='__main__':main()
