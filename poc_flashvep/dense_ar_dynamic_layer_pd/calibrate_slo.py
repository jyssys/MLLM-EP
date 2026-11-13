"""Freeze independent calibration thresholds before Dynamic evaluation."""
import fcntl,math
from .common import *
from .prepare import base
from .run import execute
from .analyze import metrics


FACTORS={'TTFT_priority':(1.25,2.,2.,2000),'Balanced':(2.,1.25,1.25,5000),'Decode_priority':(4.,1.1,1.1,10000)}


def main():
    initialize();target=ROOT/'configs/frozen_SLO_regimes.json'
    if target.exists():print('SLO_ALREADY_FROZEN',digest(target));return
    results=[];thresholds={}
    try:
        for cap in (16,32,64,128,200):
            starts=[]
            for start in range(3):
                c=base(f'slo_serial_O{cap}_N16_s{start}_v1',count=16,cap=cap,dataset='calibration.json')
                c.update(calibration_serial=True,arrival_offsets=[0.]*16,arrival_rate=None,
                    arrival_seed=2026111310+start,phase='INDEPENDENT_SLO_CALIBRATION',timeout=400)
                name=save(c)
                if not execute(name,restore=False):return
                row=metrics(name);starts.append(row);results.append(row)
                write(ROOT/'state/SLO_calibration_progress.json',dict(results=results,Dynamic_evaluated=False))
                print('SLO_CALIBRATION',cap,start,row['TTFT_p95_ms'],row['ITL_p99_ms'],flush=True)
            anchor={k:max(row[k] for row in starts) for k in ('TTFT_p95_ms','ITL_p99_ms','TPOT_p95_ms')}
            thresholds[str(cap)]={label:dict(TTFT_ms=math.ceil(max(floor,a*anchor['TTFT_p95_ms'])),
                ITL_ms=math.ceil(b*anchor['ITL_p99_ms']),TPOT_ms=math.ceil(c*anchor['TPOT_p95_ms']))
                for label,(a,b,c,floor) in FACTORS.items()}
            write(ROOT/'analysis/SLO_calibration_inputs.json',dict(results=results,output_caps=thresholds,
                decision_rule=FACTORS,Dynamic_evaluated=False,evidence='MEASURED_LOW_LOAD_CALIBRATION'))
        write(target,dict(frozen_at=time.time(),output_caps=thresholds,independent_starts_per_cap=3,
            calibration_names=[row['name'] for row in results],decision_rule=FACTORS,
            Dynamic_evaluated_before_freeze=False,
            scope='Research SLO regimes. TTFT floors were specified before Dynamic results. No candidate measurement selects a threshold. Native-relative 1.05 targets remain separate.'))
        write(ROOT/'state/SLO_freeze_receipt.json',dict(time=time.time(),path=str(target.relative_to(REPO)),SHA256=digest(target),
            Dynamic_evaluated=False,scope='Do not change thresholds after this receipt.'))
        print('SLO_FROZEN',digest(target),flush=True)
    finally:
        with QUIET_LOCK.open('a+') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX)
            if not resources().processes():resources().restore(list(range(8)))


if __name__=='__main__':main()
