"""Give the full-replica baseline the same no-existing-Decode guard."""
import fcntl,json
from .common import *
from .prepare import base,save
from .run import execute
from .analyze import compare,metrics


def main():
    jobs=[]
    for count in (8,32):
        c=base(f'colocated_qwen8_L8000_O16_full_cohort_N{count}_qual_v4',count=count,cap=16)
        c.update(arm='DP2',colocated_cohorts=True,D_admission_batch=4,D_cohort_collect_ms=2,
            batch_commands=True,atomic_prompt=True,fixed_KV_bytes=42*1024**3,
            warmup_batch_sizes=[1,2,3,4],warmup_output_cap=16,normal_lifetime_audit=True,
            phase='GENERIC_FULL_REPLICA_COHORT_CORRECTNESS')
        jobs.append(save(c))
    template=json.loads((ROOT/'configs/serving_qwen8_L8000_O16_N64_s3_native_pilot_v1.json').read_text())
    c=dict(template,name='serving_qwen8_L8000_O16_N64_s3_colocated_full_cohort_v4',arm='DP2',
        layer_mode=None,hidden_transport=None,serialize_D_imports=False,cohort_barrier=False,
        colocated_cohorts=True,normal_lifetime_audit=True,phase='GENERIC_FULL_REPLICA_COHORT_SELECTION',
        scope='Two full replicas on the same GPU budget. Each replica accepts a bounded cohort only when its existing requests finish. No Layer migration or KV transfer. This guard is also available to the generic baseline, not only the proposed method.')
    pilot=save(c)
    write(ROOT/'state/colocated_cohort_plan.json',dict(jobs=jobs,pilot=pilot))
    try:
        for n in jobs:
            if not execute(n,restore=False):return
            g=compare('native_qwen8_L8000_O32_reference32_s0',n,True)
            write(ROOT/'analysis'/(n+'_output_gate.json'),g);print('COHORT_BASELINE_GATE',n,g['passed'],flush=True)
            if not g['passed']:return
        if execute(pilot,restore=False):
            g=compare('native_qwen8_L8000_O16_N128_s4_cohort_cleanA_v4',pilot,True)
            write(ROOT/'analysis/colocated_cohort_pilot.json',dict(output_gate=g,metrics=metrics(pilot)))
    finally:
        with (ROOT/'state/quiet_lane.lock').open('a+') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX)
            if not resources().processes():resources().restore(list(range(8)))


if __name__=='__main__':main()
