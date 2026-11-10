"""Freeze reused workload assets and new immutable experiment configurations."""
import copy
import random
import shutil
from .common import *


def prepare():
    initialize()
    for path in (OLD/'data').glob('*.json'):
        target=ROOT/'data'/path.name
        if not target.exists():shutil.copy2(path,target)
        assert digest(path)==digest(target)
    slo=OLD/'configs/frozen_SLO.json';target=ROOT/'configs/frozen_SLO.json'
    if not target.exists():shutil.copy2(slo,target)
    assert digest(slo)==digest(target)
    tool_source=REPO/'poc_flashvep/reports/ouro_huginn_loop_aware_d_local_prefill_cross_phase_batching/tools'
    for path in tool_source.glob('activity_build.json'):
        if path.is_file():
            target=ROOT/'tools'/path.name;target.parent.mkdir(exist_ok=True)
            if not target.exists():shutil.copy2(path,target)


def config(model,name,method='native',count=128,seed=2026110910,trace=False):
    arm='native' if method=='native' else 'token' if method=='token' else 'tile'
    source=OLD/'configs'/f'confirm_{model}_{arm}_seed2026110911.json'
    cfg=copy.deepcopy(json.loads(source.read_text()))
    cfg.update(name=name,count=count,arrival_seed=seed,
        phase='COMPLETION_AWARE_REIMPLEMENTATION',clean_confirmation=False,
        kernel_activity_seconds=0.,sparse_device_trace=trace,sparse_decode_period=1,
        observe_device_execution=False,observe_service_timing=False,
        timeout=600,startup_timeout=300)
    cfg['prompt_ids']=cfg['prompt_ids'][:count];cfg['request_caps']=cfg['request_caps'][:count]
    cfg['data_scope']['distinct_prompt_ids']=count
    rng=random.Random(seed);offsets=[0.]
    for _ in range(count-1):offsets.append(offsets[-1]+rng.expovariate(cfg['arrival_rate']))
    cfg['arrival_offsets']=offsets
    if method=='old':cfg['use_old_scheduler']=True
    return cfg


def main():
    prepare()
    jobs=[]
    for model in ('ouro','huginn'):
        for suffix,trace in (('clean',False),('events',True)):
            name=f'baseline_{model}_{suffix}128_calibration'
            cfg=config(model,name,trace=trace);save(cfg);jobs.append(name)
    write(ROOT/'state/initial_jobs.json',jobs)
    print(json.dumps(jobs))


if __name__=='__main__':main()
