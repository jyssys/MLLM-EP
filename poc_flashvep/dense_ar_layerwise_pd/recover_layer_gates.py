"""Require real suffix work for every candidate cut before serving."""
import fcntl,json
from .common import *
from .prepare import save
from .run import execute
from .actual_layer_gate import gate


def main():
    first='layer_qwen8_L8000_O16_c35_separate_suffix_N8_recovery_v11'
    if not (ROOT/'raw'/first/'dispatcher_result.json').exists():return
    if not json.loads((ROOT/'raw'/first/'dispatcher_result.json').read_text())['complete']:return
    if not gate(first)['passed']:return
    jobs=[]
    for count in (8,32):
        for cut in (35,34,32,28):
            if count==8 and cut==35:continue
            c=json.loads((ROOT/'configs'/f'layer_qwen8_L8000_O16_migrate_c{cut}_N{count}_gpu_v6.json').read_text())
            c.update(name=f'layer_qwen8_L8000_O16_c{cut}_separate_suffix_N{count}_qual_v12',
                separate_suffix_runner=True,verify_layer_execution=True,execution_events=True,batch_commands=True,
                phase='ACTUAL_SUFFIX_EXECUTION_QUALIFICATION')
            jobs.append(save(c))
    write(ROOT/'state/actual_suffix_correctness_plan.json',dict(first=first,jobs=jobs))
    results=[dict(name=first,passed=True)]
    try:
        for n in jobs:
            if not execute(n,restore=False):break
            g=gate(n);results.append(dict(name=n,passed=g['passed']))
            write(ROOT/'state/actual_suffix_correctness_progress.json',dict(results=results))
            if not g['passed']:break
    finally:
        with (ROOT/'state/quiet_lane.lock').open('a+') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX)
            if not resources().processes():resources().restore(list(range(8)))


if __name__=='__main__':main()
