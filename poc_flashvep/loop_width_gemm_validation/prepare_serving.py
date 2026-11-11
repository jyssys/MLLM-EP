"""Prepare paired marker-free discovery. This does not start a GPU job."""
import argparse,copy,random
from .common import *


def cell(model,arm,count,cap,arrival_seed,phase,version='v2'):
    prefix='kernel_v2' if version=='v2' else 'kernel'
    cfg=json.loads((ROOT/'configs'/f'{prefix}_{model}_{arm}_state32.json').read_text())
    pool=json.loads((ROOT/'data'/cfg['dataset']).read_text())
    assert len(pool)>=count and len({r['prompt_id'] for r in pool[:count]})==count
    rate=1.971 if model=='huginn' else 6.356;rng=random.Random(arrival_seed);offsets=[0.]
    for _ in range(count-1):offsets.append(offsets[-1]+rng.expovariate(rate))
    name=f'serve_{version}_{model}_{arm}_O{cap}_N{count}_a{arrival_seed}'
    cfg.update(name=name,width_arm=arm,count=count,cap=cap,request_caps=[cap]*count,
        prompt_ids=[r['prompt_id'] for r in pool[:count]],arrival_seed=arrival_seed,
        arrival_offsets=offsets,arrival_rate=rate,arrival_pattern='OPEN_LOOP_POISSON',
        phase=phase,workload=f'L{cfg["length"]}_O{cap}',
        state_probe=False,verify_transfer=False,async_prompt_state_probe=False,
        async_verify_handoff=False,calibration_serial=False,
        observe_device_execution=False,observe_service_timing=False,
        sparse_device_trace=False,kernel_activity_seconds=0.,nsys=False,host_cost_trace=False,
        observe_P_submission_readiness=False,clean_confirmation=True,
        required_qualification=f'{prefix}_{model}_{arm}_state32',timeout=600,
        data_scope=dict(distinct_prompt_ids=count,pool_records=len(pool),cyclic_reuse=False,
            prefix_caching=False,scope='Preserved controlled pool. New arrival seed. Finite cohort, not sustainable capacity.'),
        short_output_scope='Full prompt state qualified at O16. This separate O4 discovery must also pass exact output-prefix and resource-lifetime checks.' if cap==4 else None)
    save(cfg);return name


def main():
    initialize();p=argparse.ArgumentParser();p.add_argument('--version',choices=('v1','v2'),default='v2');args=p.parse_args();jobs=[]
    for model in ('huginn','ouro'):
        arms=('zero','loop','token','token_group32','loop28','token1792') if model=='huginn' else ('zero','loop','token')
        caps=(16,4) if model=='huginn' else (32,)
        for cap in caps:
            for arm in arms:
                jobs.append(cell(model,arm,64,cap,2026111110,'MARKER_FREE_STATIC_DISCOVERY_NOT_CONFIRMATION',args.version))
    write(ROOT/'state'/f'serving_{args.version}_discovery_plan.json',dict(jobs=jobs,
        selection='Same arrival seed per workload. Final selected methods need new arrival seeds. Kernel policy stays fixed from projection start 1.'))
    print(' '.join(jobs))


if __name__=='__main__':main()
