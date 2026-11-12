"""Preregister clean, host-only, and actual-kernel matched Native controls."""
import json,random
from .common import *
from .prepare import base,save
from .activity import build


def main():
    initialize();build();plan=json.loads((ROOT/'state/discovery_plan.json').read_text())
    rate=plan['P_capacity_requests_s']*1.3;jobs=[]
    for suffix,observe,activity in (('cleanA',False,False),('host',True,False),('kernel',True,True),('cleanB',False,False)):
        rng=random.Random(2026111204);offsets=[0.]
        for _ in range(127):offsets.append(offsets[-1]+rng.expovariate(rate))
        c=base('native_qwen8_L8000_O16_N128_load130_s4_'+suffix,count=128,cap=16)
        c.update(dataset='discovery_qwen8_L8000.json',arrival_rate=rate,arrival_offsets=offsets,
            arrival_seed=2026111204,phase='NATIVE_MATCHED_OBSERVATION_CONTROL',timeout=240,
            clean_timing=not activity,runtime_observation=observe,physical_activity=activity,
            activity_roles=['prefill','decode'],activity_host_pool_bytes=1024**3,activity_memcpy=False)
        jobs.append(save(c))
    write(ROOT/'state/native_confirmation_plan.json',dict(jobs=jobs,
        scope='Ordered A/host/kernel/B comparison. Future qualification uses the 5% whole-run and local tail interference target. Kernel activity alone does not establish Decode-safe capacity.'))
    print(json.dumps(jobs))


if __name__=='__main__':main()
