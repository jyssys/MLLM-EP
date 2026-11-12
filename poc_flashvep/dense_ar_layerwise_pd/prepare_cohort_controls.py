"""Hold batch composition constant in a matched observer control."""
import json
from .common import *
from .prepare import save


def main():
    initialize();old=json.loads((ROOT/'state/quiet_confirmation_plan.json').read_text());jobs=[]
    template=json.loads((ROOT/'configs'/(old['jobs'][0]+'.json')).read_text())
    for label,observed in (('cleanA',False),('quiet',True),('cleanB',False)):
        c=dict(template,name='native_qwen8_L8000_O16_N128_s4_cohort_'+label+'_v4',
            runtime_observation=observed,runtime_observation_roles=['prefill'],physical_activity=observed,
            serialize_D_imports=True,D_admission_batch=4,D_cohort_collect_ms=2,cohort_barrier=True,
            batch_commands=True,fixed_KV_bytes=42*1024**3,
            phase='MATCHED_COHORT_OBSERVER_CONTROL',
            scope='No Layer migration. The same Native cohort admission is used in A/quiet/B. Native unguarded controls remain separate. This test checks whether changing partial-import batch composition caused observer tail variance.')
        jobs.append(save(c))
    c=json.loads((ROOT/'configs/capacity_qwen8_L8000_O16_p_only_B124_s0.json').read_text())
    c.update(name='capacity_qwen8_L8000_O16_p_only_fixed42GiB_B124_s0',fixed_KV_bytes=42*1024**3,
        jobs=[dict(batch=b,cap=16,repeats=5) for b in (1,2,4)])
    capacity=save(c)
    write(ROOT/'state/cohort_confirmation_plan.json',dict(jobs=jobs,capacity_job=capacity,
        scope='Deterministic common warm-up, fixed KV budget, no existing Decode at new-cohort admission. No GPU sleep or artificial P slowdown. Native pending and compute-ready queues remain distinct.'))
    print(json.dumps([capacity]+jobs))


if __name__=='__main__':main()
