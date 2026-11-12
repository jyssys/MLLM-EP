"""Repeat the qualified censored observer without changing its scope."""
import fcntl,json,random
from .common import *
from .prepare import save
from .run import execute
from .quiet_analysis import main as analyze


def main():
    template=json.loads((ROOT/'configs/native_qwen8_L8000_O16_N128_s4_cohort_cleanA_v4.json').read_text())
    jobs=[]
    for seed in (2026111206,2026111207):
        rng=random.Random(seed);offsets=[0.]
        for _ in range(127):offsets.append(offsets[-1]+rng.expovariate(template['arrival_rate']))
        group=[]
        for label,observed in [('cleanA',False),('quiet',True),('cleanB',False)]:
            c=dict(template,name=f'native_qwen8_L8000_O16_N128_s{seed%100}_cohort_{label}_repeat_v1',
                arrival_seed=seed,arrival_offsets=offsets,runtime_observation=observed,
                physical_activity=observed,runtime_observation_roles=['prefill'],
                phase='INDEPENDENT_CENSORED_PHYSICAL_REPLICATION')
            group.append(save(c))
        plan=f'quiet_repeat_s{seed%100}.json';write(ROOT/'state'/plan,dict(jobs=group,
            capacity_job='capacity_qwen8_L8000_O16_p_only_fixed42GiB_B124_s0'))
        jobs.append((group,plan,f'native_quiet_repeat_s{seed%100}.json'))
    try:
        for group,plan,output in jobs:
            if not all(execute(n,restore=False) for n in group):break
            analyze(plan,output)
    finally:
        with (ROOT/'state/quiet_lane.lock').open('a+') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX)
            if not resources().processes():resources().restore(list(range(8)))


if __name__=='__main__':main()
