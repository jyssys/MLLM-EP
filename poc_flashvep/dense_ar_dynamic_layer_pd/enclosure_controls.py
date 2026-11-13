"""Check CUDA event overhead separately from kernel activity profiling."""
import fcntl
from .common import *
from .run import execute
from .analyze import compare,metrics


def main():
    initialize();deadline=time.monotonic()+18000
    target=ROOT/'configs/heldout_selection.json'
    while not target.exists():
        assert time.monotonic()<deadline,'No frozen validation selection'
        time.sleep(5)
    selection=json.loads(target.read_text());results=[]
    try:
        for arm in ('Native','Colocated','Static','COST','Last8'):
            chosen=(selection['selected']['16'][arm] if arm!='Last8' else 'validation_fast_Static8_O16_N64_v1')
            template=json.loads((ROOT/'configs'/(chosen+'.json')).read_text())
            rows=[]
            for variant in ('A','events','C'):
                c=dict(template);c.update(name=f'enclosure_{arm}_O16_N64_{variant}_v1',
                    execution_events=variant=='events',phase='CUDA_EVENT_ENCLOSURE_OVERHEAD_CONTROL_NOT_KERNEL_UNION')
                if not execute(save(c),restore=False):return
                row=dict(name=c['name'],metrics=metrics(c['name']),
                    output_gate=compare('coarse_O16_N64_q4.777_native16_s1_v1',c['name']))
                rows.append(row)
                if not row['output_gate']['passed']:return
            a,b,c=(r['metrics'] for r in rows)
            keys=('TTFT_p95_ms','ITL_p99_ms','TPOT_p95_ms','E2E_p95_ms','maximum_ITL_ms')
            ratios={k:b[k]/((a[k]+c[k])/2) for k in keys};aa={k:c[k]/a[k] for k in keys}
            result=dict(arm=arm,rows=rows,profile_to_mean_controls=ratios,A_A_ratio=aa,
                interference_gate_passed=all(.95<=r<=1.05 for r in (*ratios.values(),*aa.values())),
                scope='CUDA events measure stream enclosures. They do not measure exclusive kernels, physical idle, or SM headroom.')
            results.append(result);write(ROOT/'state/enclosure_controls_progress.json',dict(results=results))
            print('ENCLOSURE_CONTROL',arm,result['interference_gate_passed'],flush=True)
        write(ROOT/'analysis/enclosure_control_results.json',dict(results=results))
    finally:
        with QUIET_LOCK.open('a+') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX)
            if not resources().processes():resources().restore(list(range(8)))


if __name__=='__main__':main()
