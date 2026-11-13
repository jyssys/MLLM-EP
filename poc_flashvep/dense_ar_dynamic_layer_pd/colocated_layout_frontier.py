"""Allow Colocated to keep its best legal layout and cohort size."""
import fcntl
from .common import *
from .prepare import base
from .run import execute
from .analyze import compare,metrics


def main():
    initialize();results=[]
    try:
        for cap,maximum,layout in ((16,4,'NHD'),(16,4,'HND'),(16,16,'NHD'),(32,16,'NHD'),(64,16,'NHD')):
            c=base(f'frontier_coloc_O{cap}_B{maximum}_{layout}_N64_v1',arm='coloc_cohort',cap=cap,rate=4.777100014491616)
            c.update(hidden_transport=None,kv_layout=layout,D_admission_batch=maximum,
                token_budget=maximum*8000,D_slots=16,warmup_batch_sizes=[1,2,4,8] if maximum>4 else [1,2,4],
                phase='INDEPENDENT_COLOCATED_LAYOUT_AND_COHORT_TUNING')
            if not execute(save(c),restore=False):return
            row=dict(name=c['name'],output_gate=compare(f'coarse_O{cap}_N64_q4.777_native16_s1_v1',c['name']),metrics=metrics(c['name']))
            results.append(row);write(ROOT/'state/colocated_layout_frontier_progress.json',dict(results=results))
            print('COLOCATED_FRONTIER',c['name'],row['output_gate']['passed'],row['metrics']['TTFT_p95_ms'],row['metrics']['ITL_p99_ms'],flush=True)
        write(ROOT/'analysis/colocated_layout_frontier_results.json',dict(results=results,
            scope='Native NIXL requires HND. Colocated may choose NHD or HND. Matching layouts is a diagnostic control, not a restriction on the strongest baseline.'))
    finally:
        with QUIET_LOCK.open('a+') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX)
            if not resources().processes():resources().restore(list(range(8)))


if __name__=='__main__':main()
