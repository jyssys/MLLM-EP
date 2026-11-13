"""Retain external activity and marker-free A/B/A latency controls."""
import fcntl
from .common import *
from .configuration import base
from .run import execute
from .analyze import compare,metrics


def main():
    rows=[]
    try:
        for variant in ('A','capture','C'):
            cfg=base(f'nsys_bounded_Native_O16_N64_{variant}_v1',cap=16,
                nsys_bounded_capture=variant=='capture',nsys_seconds=2.,nsys_offset_s=5.,
                phase='EXTERNAL_BOUNDED_ACTIVITY_QUALIFICATION')
            name=save(cfg)
            if not execute(name,restore=False):
                write(ROOT/'state/nsys_check_progress.json',dict(results=rows,failed=name));break
            gate=compare('baseline_Native_O200_N64_q4p8_s21_v1',name,True)
            row=dict(name=name,output=gate,metrics=metrics(name));rows.append(row)
            write(ROOT/'state/nsys_check_progress.json',dict(results=rows))
            print('NSYS_CONTROL',name,gate['passed'],flush=True)
            if not gate['passed']:break
        result=dict(results=rows,passed=False,physical_idle='NOT_MEASURED')
        if len(rows)==3:
            a,b,c=(r['metrics'] for r in rows)
            keys=('TTFT_p95_ms','ITL_p99_ms','TPOT_p95_ms','E2E_p95_ms','maximum_ITL_ms')
            result['capture_to_mean_controls']={k:b[k]/((a[k]+c[k])/2) for k in keys}
            result['A_A_ratios']={k:c[k]/a[k] for k in keys}
            result['interference_passed']=all(.95<=v<=1.05 for group in ('capture_to_mean_controls','A_A_ratios') for v in result[group].values())
        result['scope']='External activity is diagnostic until independent coverage and interference gates pass. Event enclosures and NVML are not physical kernel idle.'
        write(ROOT/'analysis/nsys_bounded_interference.json',result)
    finally:
        with QUIET_LOCK.open('a+') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX)
            if not resources().processes():resources().restore(list(range(8)))


if __name__=='__main__':main()
