"""Freeze source data and existing SLOs before adaptive runtime experiments."""
import copy
import random
from .common import *


def put(cfg,count,rate=None,split='test'):
    data=ROOT/'data'/f'{split}_{cfg["model"]}_L{cfg["length"]}.json'
    rows=json.loads(data.read_text());assert count<=len(rows)
    cfg.update(count=count,dataset=data.name,request_caps=[cfg['cap']]*count,
        prompt_ids=[r['prompt_id'] for r in rows[:count]],arrival_offsets=[0.]*count)
    if rate is not None:
        rng=random.Random(cfg['arrival_seed'])
        for i in range(1,count):cfg['arrival_offsets'][i]=cfg['arrival_offsets'][i-1]+rng.expovariate(rate)
        cfg.update(arrival_rate=rate,arrival_pattern='OPEN_LOOP_POISSON')
    else:cfg.pop('arrival_rate',None)
    cfg['atomic_prompt']=cfg['chunk']>=cfg['length'];cfg['chunk_label']='full' if cfg['atomic_prompt'] else str(cfg['chunk'])
    assert cfg['length']+max(cfg['request_caps'])<=cfg['context_limit']
    path=ROOT/'configs'/(cfg['name']+'.json')
    if path.exists():assert json.loads(path.read_text())==cfg
    else:write(path,cfg)
    return cfg['name']


def base(model,name,length=None,cap=None,**extra):
    old='warm_secondary_ouro_l8192_c8192_q2p0_pilot24' if model=='ouro' else 'warm_epoch_huginn_c2048_q1p2_pilot24'
    cfg=copy.deepcopy(json.loads((PREVIOUS/'configs'/(old+'.json')).read_text()))
    cfg.update(name=name,model=model,phase='autonomous_pilot',seed=2026110307,arrival_seed=2026110311,
        gpus=[0,1] if model=='ouro' else [4,5],timeout=300,startup_timeout=240,
        observe_descriptor_cost=False,record_descriptor_replay=False,
        verify_transfer=False,state_probe=False,nsys=False,cuda_only_profile=False,P_only_profiler=False,
        measurement_epoch='AUTONOMOUS_V1',prime_all_graph_shapes=True,async_evidence=True)
    cfg.pop('capture_roles',None)
    if length is not None:
        cfg.update(length=length,state_capacity_tokens=length,chunk=length,
            token_budget=max(length,(32768//length)*length),workload=f'L{length}_O{cap or cfg["cap"]}')
    if cap is not None:cfg['cap']=cap
    cfg.update(extra)
    return cfg


def main():
    initialize()
    corpus=json.loads((ATOMIC/'data/long_source_pool.json').read_text())
    for model,lengths in {'ouro':[4096,6144,8192], 'huginn':[1536,2048,2560,3072,3500,3900]}.items():
        for length in lengths:
            for split,records in [('calibration',corpus[:24]),('test',corpus[24:48]),('confirmation',corpus[48:176])]:
                rows=[dict(prompt_id=r['prompt_id'],source_SHA256=r['source_SHA256'],source_dataset=r['source_dataset'],
                    input_ids={model:r['input_ids'][model][:length]},length=length,
                    source_type='LENGTH_MATCHED_CONTROLLED_PUBLIC_TEXT_PREFIX',truncation_explicit=True)
                    for r in records if len(r['input_ids'][model])>=length]
                assert len(rows)==len(records),'The pinned corpus must support the actual selected context'
                write(ROOT/'data'/f'{split}_{model}_L{length}.json',rows)
    frozen=json.loads((PREVIOUS/'configs/frozen_SLO.json').read_text())
    write(ROOT/'configs/frozen_SLO.json',frozen)
    write(ROOT/'state/frozen_SLO_import.json',dict(time=time.time(),source=str(PREVIOUS/'configs/frozen_SLO.json'),
        source_SHA256=digest(PREVIOUS/'configs/frozen_SLO.json'),
        scope='Existing thresholds are unchanged. A new input length must calibrate before its first held-out pilot.'))
    write(ROOT/'state/data_provenance.json',dict(source=str(ATOMIC/'data/long_source_pool.json'),SHA256=digest(ATOMIC/'data/long_source_pool.json'),
        calibration_indices=[0,24],pilot_indices=[24,48],confirmation_indices=[48,176],
        scope='Current pilot and final prompts are disjoint. This is the previously acquired public corpus, not a new-domain evaluation. Controlled prefixes and ignore-EOS caps are explicit.'))
    names=[]
    for model in ('ouro','huginn'):
        c=base(model,'baseline_'+model+'_pilot24')
        names.append(put(c,24,2.5 if model=='ouro' else 1.0))
    write(ROOT/'state/initial_candidates.json',dict(time=time.time(),baseline=names,
        scope='Only starting references. The next candidates depend on actual gap and correctness evidence. No blind grid is scheduled.'))
    print(json.dumps(dict(baselines=names,hard_deadline=DEADLINE)))


if __name__=='__main__':main()
