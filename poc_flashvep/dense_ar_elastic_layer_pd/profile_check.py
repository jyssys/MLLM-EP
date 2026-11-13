"""Try one warmed background capture and retain both marker-free controls."""
import fcntl
from .common import *
from .configuration import base
from .run import execute
from .analyze import compare,metrics


def main():
    from .activity import build
    initialize();build();rows=[]
    try:
        for variant in ('A','capture','C'):
            cfg=base(f'background_profile_Native_O16_N64_{variant}_v1',cap=16,
                activity_background=variant=='capture',activity_background_seconds=.6,
                activity_background_offset_s=5.,activity_roles=['prefill','decode'],activity_memcpy=True,
                phase='BOUNDED_BACKGROUND_CUPTI_INTERFERENCE_CONTROL_NOT_PRIMARY')
            if not execute(save(cfg),restore=False):break
            output=compare('baseline_Native_O200_N64_q4p8_s21_v1',cfg['name'],True)
            rows.append(dict(name=cfg['name'],output=output,metrics=metrics(cfg['name'])))
            write(ROOT/'state/background_profile_progress.json',dict(results=rows))
            if not output['passed']:break
        result=dict(results=rows,passed=False)
        if len(rows)==3:
            a,b,c=(r['metrics'] for r in rows);keys=('TTFT_p95_ms','ITL_p99_ms','TPOT_p95_ms','E2E_p95_ms','maximum_ITL_ms')
            result['capture_to_mean_controls']={k:b[k]/((a[k]+c[k])/2) for k in keys}
            result['A_A_ratios']={k:c[k]/a[k] for k in keys}
            result['passed']=all(.95<=v<=1.05 for group in ('capture_to_mean_controls','A_A_ratios') for v in result[group].values())
        result['scope']='One diagnostic attempt. Failed interference gates cannot establish clean kernel busy or idle. Do not substitute event enclosures.'
        write(ROOT/'analysis/background_profile_interference.json',result);print('BACKGROUND_PROFILE_GATE',result['passed'],flush=True)
    finally:
        with QUIET_LOCK.open('a+') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX)
            if not resources().processes():resources().restore(list(range(8)))


if __name__=='__main__':main()
