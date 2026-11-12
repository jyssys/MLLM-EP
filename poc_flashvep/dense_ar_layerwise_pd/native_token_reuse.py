"""Qualify exact source-token reuse before treating it as a stronger control."""
import fcntl,json
from .common import *
from .prepare import save
from .run import execute
from .analyze import compare,metrics


def main():
    jobs=[]
    for count in (8,32):
        c=json.loads((ROOT/'configs'/f'eager_qwen8_L8000_O16_N{count}_reference_qual_v16.json').read_text())
        c.update(name=f'eager_qwen8_L8000_O16_N{count}_native_token_reuse_qual_v19',
            reuse_prefill_token=True,state_capture=False,execution_events=False,normal_lifetime_audit=True,
            clean_timing=False,phase='SOURCE_FIRST_TOKEN_NATIVE_PROTOCOL_QUALIFICATION')
        jobs.append(save(c))
    write(ROOT/'state/native_token_reuse_plan.json',dict(jobs=jobs,release_policy='After Native import and next full Decode forward host submission. No GPU completion prediction. All holding time remains in TTFT.'))
    results=[]
    try:
        for n in jobs:
            c=json.loads((ROOT/'configs'/(n+'.json')).read_text());count=c['count']
            if not execute(n,restore=False):return
            check=compare(f'eager_qwen8_L8000_O16_N{count}_reference_qual_v16',n)
            life={role:json.loads((ROOT/'raw'/n/f'KV_lifetime_{role}.json').read_text()) for role in ('prefill','decode')}
            passed=check['passed'] and all(x['passed'] for x in life.values())
            row=dict(name=n,passed=passed,output_check=check,lifetime=life,metrics=metrics(n))
            results.append(row);write(ROOT/'state/native_token_reuse_progress.json',dict(results=results))
            print('TOKEN_REUSE_GATE',n,passed,flush=True)
            if not passed:return
    finally:
        with (ROOT/'state/quiet_lane.lock').open('a+') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX)
            if not resources().processes():resources().restore(list(range(8)))


if __name__=='__main__':main()
