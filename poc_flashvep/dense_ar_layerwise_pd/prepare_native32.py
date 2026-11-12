"""Freeze paired N32 correctness and independent capacity cells."""
import json,random,time
from .common import *
from .prepare import base,save


def main():
    initialize();gate=json.loads((ROOT/'analysis/native_initial_qualification.json').read_text());assert gate['passed']
    rng=random.Random(2026111202);offsets=[0.]
    for _ in range(31):offsets.append(offsets[-1]+rng.expovariate(2.))
    jobs=[]
    for label in ('reference','repeat','dp2','chunk2048'):
        c=base('native_qwen8_L8000_O32_'+label+'32_s0',count=32)
        c.update(phase='NATIVE_N32_CORRECTNESS',arrival_rate=2.,arrival_seed=2026111202,
            arrival_offsets=offsets,timeout=300,environment_revision='EXPLICIT_NATIVE_PYTHON_BIN_PATH')
        if label=='dp2':c.update(arm='DP2',atomic_prompt=False)
        if label=='chunk2048':c.update(chunk=2048,atomic_prompt=False)
        jobs.append(save(c))
    for mode in ('P_ONLY','PREPARED_D'):
        c=base('capacity_qwen8_L8000_O16_'+mode.lower()+'_B124_s0',count=8,cap=16)
        c.update(module='capacity',phase='INDEPENDENT_NATIVE_CAPACITY',gpus=[0],
            capacity_mode=mode,jobs=[dict(batch=b,cap=16,repeats=3) for b in (1,2,4)],
            timeout=300,clean_timing=True,environment_revision='EXPLICIT_NATIVE_PYTHON_BIN_PATH')
        jobs.append(save(c))
    write(ROOT/'state/native32_capacity_plan.json',dict(time=time.time(),jobs=jobs,
        scope='N32 exact generated output controls precede Layer work. Independent Local service measurements guide QPS only. They do not establish online headroom.'))
    print(json.dumps(jobs))


if __name__=='__main__':main()
