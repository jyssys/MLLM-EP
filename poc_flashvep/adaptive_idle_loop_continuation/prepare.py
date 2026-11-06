"""Freeze paired open-loop requests before candidate measurements."""
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


def configuration(model,name,count=128,load='heavy',seed=2026110611,observer='off',state=False):
    cfg=copy.deepcopy(json.loads((PREVIOUS/'configs'/f'local_{model}_native_n128_s2026110701_off.json').read_text()))
    cfg.update(name=name,count=count,arrival_seed=seed,arrival_rate=RATES[model][load],
        phase='ADAPTIVE_IDLE_GATED_CONTINUATION',idle_campaign=True,idle_observer=observer,
        dlocal_campaign=False,dlocal_method='native',dlocal_activity_thin=False,dlocal_model_events=False,
        dlocal_service_once=False,dlocal_service_pool=False,freeze_dlocal_sources=False,
        state_probe=state,verify_transfer=state,observe_service_timing=False,
        native_path=False,feature_native_wire_v5=False,load=load,
        timeout=900 if state else 600,startup_timeout=300)
    pool=json.loads((ROOT/'data'/cfg['dataset']).read_text())
    assert count<=len(pool),'No implicit prompt cycling'
    cfg['prompt_ids']=[v['prompt_id'] for v in pool[:count]];cfg['request_caps']=[cfg['cap']]*count
    rng=random.Random(seed);cfg['arrival_offsets']=[0.]
    for _ in range(count-1):cfg['arrival_offsets'].append(cfg['arrival_offsets'][-1]+rng.expovariate(cfg['arrival_rate']))
    cfg['data_scope']=dict(pool_records=len(pool),measured_records=count,cyclic_reuse=False,
        selection='Preserved ShareGPT/LongBench controlled-prefix pool. Not unseen-text confirmation.')
    return cfg


if __name__=='__main__':prepare()
