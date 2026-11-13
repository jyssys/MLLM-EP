"""Remove the unused Hidden agent from the longer-output Native anchors."""
import fcntl
from .common import *
from .run import execute
from .analyze import metrics,compare


def main():
    initialize();rows=[]
    try:
        for cap in (128,200):
            name=f'coarse_O{cap}_N64_q4.777_native16_s1_v1'
            c=json.loads((ROOT/'configs'/(name+'.json')).read_text())
            c.update(name=f'lean_native_O{cap}_N64_v1',kv_layout='HND',hidden_transport=None,
                phase='LONG_OUTPUT_NATIVE_ANCHOR_WITHOUT_UNUSED_HIDDEN_AGENT')
            if not execute(save(c),restore=False):return
            row=dict(name=c['name'],metrics=metrics(c['name']),output_gate=compare(name,c['name']))
            rows.append(row);write(ROOT/'state/native_long_output_progress.json',dict(results=rows))
            print('LONG_OUTPUT_NATIVE',c['name'],row['output_gate']['passed'],flush=True)
            if not row['output_gate']['passed']:return
        write(ROOT/'analysis/native_long_output_results.json',dict(results=rows))
    finally:
        with QUIET_LOCK.open('a+') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX)
            if not resources().processes():resources().restore(list(range(8)))


if __name__=='__main__':main()
