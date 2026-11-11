"""Test stronger observed P pressure without altering the model or SLO."""
import json,random
from .common import *


def main():
    initialize();parent='memory_ouro_L16384_O16_P2_mu115_N64_a2026111230'
    cfg=json.loads((ROOT/'configs'/(parent+'.json')).read_text())
    calibration=json.loads((ROOT/'data'/'natural_calibration_ouro_L16384.json').read_text())
    pool=[calibration[i%24] for i in range(64)];path=ROOT/'data'/'pressure_ouro_L16384_calibration24_repeated_pilot64.json'
    if path.exists():assert json.loads(path.read_text())==pool
    else:write(path,pool)
    seed=2026111280;rate=1.5;rng=random.Random(seed);offsets=[0.]
    for _ in range(63):offsets.append(offsets[-1]+rng.expovariate(rate))
    cfg.update(dataset=path.name,prompt_ids=[r['prompt_id'] for r in pool],arrival_seed=seed,
        arrival_rate=rate,arrival_offsets=offsets,
        phase='OBSERVED_SERVICE_GUIDED_NATIVE_PRESSURE_DISCOVERY',
        selection_reason='The 1.1949810287850386 QPS pilot had finite P service near 1.17 requests/s and weak queue growth. Offered 1.5 QPS is approximately 1.28 times that observed service, comparable to the original controlled-overload ratio. This is not a sustainable-load claim.',
        data_scope=dict(distinct_prompt_ids=24,requests=64,prefix_caching=False,
            scope='Calibration-only repeated pilot. Not 64 independent documents. No filler or synthetic context extension.'))
    jobs=[]
    for mode in ('clean','events','quiet_epochs','repeat'):
        c=dict(cfg);name=f'pressure_ouro_L16384_O16_P2_q150_pilot64_a{seed}_{mode}'
        c.update(name=name,comparison_reference=None,
            required_qualification='bounded_quiet_ouro_L16384_observer_state8' if mode=='quiet_epochs' else 'async_ouro_L16384_native_reference_state32',
            discovery_event_observer=mode in ('events','quiet_epochs'),discovery_activity=mode=='quiet_epochs',
            activity_quiescent_decode=True,activity_delayed_start=True,activity_device_buffers=True,
            activity_disable_after_outputs=False,activity_quiet_stop_before_submit=True,
            activity_memcpy=True,activity_window_fraction=.35,activity_window_seconds=8.,
            discovery_cupti_preload='/usr/local/cuda-12.8/targets/x86_64-linux/lib/libcupti.so.2025.1.1',
            discovery_cupti_preload_SHA256='fccdc5fcc7c32f8cf4b3ed9b17f7f187a185a6fa0a5bcc06e3434cbd1687c808')
        save(c);jobs.append(name)
    write(ROOT/'state/pressure_quiescent_plan.json',dict(clean_reference=jobs[0],jobs=jobs[1:],all_jobs=jobs,
        scope='Prospective new workload. Same frozen interference screens, including local Decode tail. Partial D coverage is not a complete idle ratio. Independent N128 starts remain required.'))
    print(' '.join(jobs))


if __name__=='__main__':main()
