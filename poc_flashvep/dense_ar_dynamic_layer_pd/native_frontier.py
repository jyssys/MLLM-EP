"""Tune the Native cohort limit before the held-out baseline is frozen."""
import fcntl
from .common import *
from .run import execute
from .analyze import metrics,compare


def main():
    initialize();rows=[]
    try:
        for cap in (16,32,64):
            template=json.loads((ROOT/'configs'/f'lean_native_O{cap}_N64_v1.json').read_text())
            for batch in (4,8):
                c=dict(template);c.update(name=f'frontier_Native_O{cap}_D{batch}_N64_v1',
                    D_admission_batch=batch,phase='STRONG_NATIVE_COHORT_VALIDATION')
                if not execute(save(c),restore=False):return
                row=dict(name=c['name'],metrics=metrics(c['name']),
                    output_gate=compare(f'coarse_O{cap}_N64_q4.777_native16_s1_v1',c['name']))
                rows.append(row);write(ROOT/'state/native_frontier_progress.json',dict(results=rows))
                print('NATIVE_FRONTIER',c['name'],row['output_gate']['passed'],row['metrics']['TTFT_p95_ms'],row['metrics']['ITL_p99_ms'],flush=True)
                if not row['output_gate']['passed']:return
        write(ROOT/'analysis/native_frontier_results.json',dict(results=rows))
    finally:
        with QUIET_LOCK.open('a+') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX)
            if not resources().processes():resources().restore(list(range(8)))


if __name__=='__main__':main()
