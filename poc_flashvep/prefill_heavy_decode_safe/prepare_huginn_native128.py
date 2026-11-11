"""Retain independent marker-free starts for the qualified Huginn pilot."""
import json,random
from .common import *


def main():
    initialize();parent='reserve_huginn_L3072_O16_P2_mu115_N64_a2026111230'
    source=json.loads((ROOT/'configs'/(parent+'.json')).read_text())
    pool=json.loads((ROOT/'data'/source['dataset']).read_text());assert len({r['prompt_id'] for r in pool[:128]})==128
    jobs=[]
    for seed in (2026111290,2026111291,2026111292):
        rng=random.Random(seed);offsets=[0.]
        for _ in range(127):offsets.append(offsets[-1]+rng.expovariate(source['arrival_rate']))
        cfg=dict(source);name=f'confirm_huginn_L3072_O16_P2_native_N128_a{seed}'
        cfg.update(name=name,count=128,prompt_ids=[r['prompt_id'] for r in pool[:128]],
            request_caps=[16]*128,arrival_seed=seed,arrival_offsets=offsets,
            required_qualification='reserve_huginn_L3072_native_reference_state32',
            phase='INDEPENDENT_NATIVE_MARKER_FREE_N128_CONFIRMATION',
            discovery_activity=False,discovery_event_observer=False,
            discovery_cupti_preload='/usr/local/cuda-12.8/targets/x86_64-linux/lib/libcupti.so.2025.1.1',
            discovery_cupti_preload_SHA256='fccdc5fcc7c32f8cf4b3ed9b17f7f187a185a6fa0a5bcc06e3434cbd1687c808',
            data_scope=dict(distinct_prompt_ids=128,requests=128,prefix_caching=False,
                scope='Independent arrival seeds after the N64 pilot. The prompt pool is not an unseen-text generalization test.'),
            comparison_reference=None)
        save(cfg);jobs.append(name)
    write(ROOT/'state/huginn_native128_plan.json',dict(jobs=jobs,
        scope='Three independent Native process starts. Same model, legal pools, and frozen SLO. Not yet three qualified physical timelines or a Loop performance result.'))
    print(' '.join(jobs))


if __name__=='__main__':main()
