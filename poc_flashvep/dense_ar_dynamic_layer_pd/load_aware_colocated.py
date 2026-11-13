"""Check a stronger routing control with current outstanding requests only."""
import fcntl
from .common import *
from .run import execute
from .analyze import metrics,compare


def main():
    initialize();selection=json.loads((ROOT/'state/load_refinement_selection.json').read_text())['selected']
    results=[]
    try:
        for cap in (16,32,64):
            c=json.loads((ROOT/'configs'/(selection[str(cap)]+'.json')).read_text())
            c.update(name=f'LA_strong_coloc_O{cap}_N64_v1',placement='LEAST_OUTSTANDING',
                phase='STRONG_COLOCATED_OBSERVED_LOAD_ROUTING_VALIDATION')
            if not execute(save(c),restore=False):return
            ref=f'coarse_O{cap}_N64_q4.777_native16_s1_v1'
            row=dict(name=c['name'],output_gate=compare(ref,c['name']),metrics=metrics(c['name']))
            results.append(row);write(ROOT/'state/load_aware_colocated_progress.json',dict(results=results))
            print('LOAD_AWARE_COLOCATED',c['name'],row['output_gate']['passed'],row['metrics']['TTFT_p95_ms'],flush=True)
        write(ROOT/'analysis/load_aware_colocated_results.json',dict(results=results,
            scope='Routing uses current local pending and active counts. It does not use future arrivals or future completion times.'))
    finally:
        with QUIET_LOCK.open('a+') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX)
            if not resources().processes():resources().restore(list(range(8)))


if __name__=='__main__':main()
