"""Build immutable request traces and model-specific experiment configurations."""
import argparse
import math
import random
from .common import *


def finish(cfg,count=None,rate=None,split='test',start=0):
    if count is not None:cfg['count']=count
    n=cfg['count'];m=cfg['model'];length=cfg['length']
    pool=cfg.get('dataset_pool',f'{split}_{m}_L{length}.json')
    assert Path(pool).name==pool,'Dataset pool must be an explicit local basename'
    source=ROOT/'data'/pool;rows=json.loads(source.read_text());pool_records=len(rows)
    assert rows,'No legal data for this input length'
    if n>len(rows):
        expanded=ROOT/'data'/(source.stem+f'_cyclic_n{n}.json')
        values=[dict(rows[i%len(rows)],source_pool_index=i%len(rows),offered_index=i) for i in range(n)]
        if expanded.exists():assert json.loads(expanded.read_text())==values
        else:write(expanded,values)
        source=expanded;rows=values
    cfg['dataset']=source.name;cfg['prompt_ids']=[r['prompt_id'] for r in rows[:n]]
    cfg['request_caps']=[cfg['cap']]*n
    cfg['data_scope']=dict(original_pool_records=pool_records,offered_count=n,explicit_cyclic_reuse=n>pool_records)
    cfg['atomic_prompt']=cfg['chunk']>=length
    cfg['independent_start']=start;cfg['arrival_seed']=cfg.get('arrival_seed_base',20261101)+start
    if rate is not None:
        cfg['arrival_rate']=rate;rng=random.Random(cfg['arrival_seed']);offsets=[0.]
        for _ in range(n-1):offsets.append(offsets[-1]+rng.expovariate(rate))
        cfg['arrival_offsets']=offsets
    else:cfg['arrival_offsets']=[0.]*n
    assert len(cfg['arrival_offsets'])==len(cfg['prompt_ids'])==len(cfg['request_caps'])==n
    if m=='qwen8':cfg.update(P_slots=4,D_slots=32,capacity=32,slots=32)
    return cfg


def initial():
    names=[]
    for model in ('qwen8','ouro','huginn'):
        for arm in ('PD_G1','DP2'):
            if model!='qwen8' and arm=='DP2':continue
            cfg=base(model,f'gate_{model}_native_full_l512_{arm.lower()}_v1',512,16,arm=arm,phase='correctness',
                verify_transfer=model!='qwen8',state_probe=model!='qwen8')
            names.append(save_config(finish(cfg,2)))
        if model!='qwen8':
            cfg=base(model,f'gate_{model}_native_chunk256_l512_v1',512,16,chunk=256,chunk_label='256',phase='correctness',
                verify_transfer=True,state_probe=True)
            names.append(save_config(finish(cfg,2)))
    write(ROOT/'configs/initial_gate_plan.json',dict(configs=names))
    print('INITIAL_GATES',len(names))


def main():
    p=argparse.ArgumentParser();p.add_argument('action',choices=('initial',));a=p.parse_args()
    if a.action=='initial':initial()


if __name__=='__main__':main()
