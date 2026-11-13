"""Compare a bounded CUDA capture with two marker-free starts."""
import argparse,fcntl
from .common import *
from .prepare import base
from .run import execute
from .analyze import metrics,compare


def main():
    p=argparse.ArgumentParser();p.add_argument('--version',type=int,default=1);p.add_argument('--duration',type=float,default=4.)
    a=p.parse_args()
    initialize();results=[]
    try:
        for variant in ('A','capture','C'):
            c=base(f'sparse_Native_O32_N64_{variant}_v{a.version}',cap=32,rate=4.777100014491616)
            c.update(kv_layout='HND',hidden_transport=None,D_admission_batch=16,
                physical_activity=variant=='capture',boundary_events=variant=='capture',
                activity_sparse_seconds=a.duration,activity_sparse_offset_s=5.,activity_memcpy=True,
                execution_events=variant=='capture',
                phase='MATCHED_SPARSE_PROFILE_INTERFERENCE_CONTROL')
            if not execute(save(c),restore=False):break
            row=dict(name=c['name'],metrics=metrics(c['name']))
            row['output_gate']=compare('coarse_O32_N64_q4.777_native16_s1_v1',c['name'])
            results.append(row);write(ROOT/f'state/sparse_profile_progress_v{a.version}.json',dict(results=results))
        write(ROOT/f'analysis/sparse_profile_control_v{a.version}.json',dict(results=results,
            scope='Independent starts. A/A variability and all metric deltas require analysis before any Clean Capacity claim.'))
    finally:
        with QUIET_LOCK.open('a+') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX)
            if not resources().processes():resources().restore(list(range(8)))


if __name__=='__main__':main()
