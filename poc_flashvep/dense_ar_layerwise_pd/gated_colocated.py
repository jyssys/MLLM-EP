import fcntl,json
from .common import *
from .run import execute
from .analyze import compare


def main():
    plan=json.loads((ROOT/'state/colocated_qualification_plan.json').read_text());results=[];failed=set()
    try:
        for name in plan['jobs']:
            c=json.loads((ROOT/'configs'/(name+'.json')).read_text())
            if c['count']==32 and c['chunk'] in failed:continue
            if not execute(name,restore=False):
                results.append(dict(name=name,passed=False,gate='GPU_EXECUTION'));break
            check=compare('native_qwen8_L8000_O32_reference32_s0',name,True)
            row=dict(name=name,passed=check['passed'],output_check=check,
                state_scope='Built-in Native chunked forward. Full per-layer state is not captured in this baseline qualification.')
            write(ROOT/'analysis'/(name+'_output_gate.json'),row);results.append(row)
            write(ROOT/'state/colocated_qualification_progress.json',dict(results=results))
            print('COLOCATED_GATE',name,row['passed'],flush=True)
            if not row['passed']:failed.add(c['chunk'])
    finally:
        with (ROOT/'state/quiet_lane.lock').open('a+') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX)
            if not resources().processes():resources().restore(list(range(8)))


if __name__=='__main__':main()
