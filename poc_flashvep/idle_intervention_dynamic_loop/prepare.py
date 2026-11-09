"""Freeze paired requests before calibration and policy testing."""
import copy
import random
import shutil
from .common import *

RATES={'ouro':dict(low=1.23460875,medium=3.4569045,heavy=6.356),
       'huginn':dict(low=.37903355455,medium=1.06129395274,heavy=1.971)}


def prepare():
    initialize()
    for model,length in (('ouro',4096),('huginn',2048)):
        source=PREVIOUS/'data'/f'heldout_{model}_L{length}.json'
        target=ROOT/'data'/source.name
        if not target.exists():shutil.copy2(source,target)
        assert digest(source)==digest(target)
    source=PREVIOUS/'configs/frozen_SLO.json';target=ROOT/'configs/frozen_SLO.json'
    if not target.exists():shutil.copy2(source,target)
    assert digest(source)==digest(target)


def configuration(model,name,count=24,load='heavy',seed=2026110801,state=False,method='native'):
    source=PREVIOUS/'configs'/f'q_{model}_native_state8.json'
    cfg=copy.deepcopy(json.loads(source.read_text()))
    cfg.update(name=name,count=count,arrival_seed=seed,arrival_rate=RATES[model][load],load=load,
        phase='IDLE_INTERVENTION_DYNAMIC_LOOP',intervention_campaign=True,
        intervention_method=method,idle_campaign=False,idle_observer='off',
        dlocal_campaign=False,D_local_probe_output_pacing=False,
        state_probe=state,verify_transfer=state,observe_service_timing=False,
        observe_device_execution=False,calibration_serial=False,
        native_path=method!='native',feature_native_wire_v5=method!='native',
        atomic_prompt=method=='native',
        timeout=900 if state else 600,startup_timeout=300)
    cfg.pop('native_request_cuts',None);cfg.pop('native_partial_cut',None)
    cfg.pop('native_frontier',None)
    pool=json.loads((ROOT/'data'/cfg['dataset']).read_text())
    assert count<=len(pool),'No implicit prompt cycling'
    cfg['prompt_ids']=[v['prompt_id'] for v in pool[:count]];cfg['request_caps']=[cfg['cap']]*count
    rng=random.Random(seed);cfg['arrival_offsets']=[0.]
    for _ in range(count-1):cfg['arrival_offsets'].append(cfg['arrival_offsets'][-1]+rng.expovariate(cfg['arrival_rate']))
    if state:
        cfg['arrival_offsets']=[0.]*count;cfg['arrival_pattern']='SIMULTANEOUS_QUALIFICATION'
        cfg['arrival_rate']=None
    cfg['data_scope']=dict(distinct_prompt_ids=count,pool_records=len(pool),cyclic_reuse=False,
        scope='Preserved controlled ShareGPT/LongBench prefix pool. No new natural-workload claim.')
    return cfg


if __name__=='__main__':
    prepare()
    for model in ('ouro','huginn'):
        save(configuration(model,f'a_{model}_native_state8',8,state=True))
        save(configuration(model,f'a_{model}_native_heavy24_s1'))
