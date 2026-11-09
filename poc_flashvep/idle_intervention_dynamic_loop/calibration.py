"""Disjoint-text device calibration and exact Token protocol cells."""
from .cells import *


def token_cell(model,dynamic=False,count=8):
    cfg=intervention(model,128 if model=='ouro' else 32,count=count,dynamic=False)
    method='dynamic_token' if dynamic else 'static_token'
    cfg.update(name=f'c_{model}_{method}_state{count}',intervention_method=method)
    cfg.pop('native_partial_cut',None)
    first=cfg['length']-512 if dynamic else 3072 if model=='ouro' else 1984
    cfg['native_frontier']=[0,first]
    cfg['remaining_cost_ms']={str(v):float(v) for v in (32,64,128,256,512)}
    cfg['boundary_checkpoints']=[cfg['length']-v for v in (512,256,128,64,32)]
    return cfg


def primitive(model,width,loops=1):
    cfg=intervention(model,width,state=False)
    cfg.update(name=f'p_{model}_tile{width}_loops{loops}_serial8',count=8,
        calibration_serial=True,observe_device_execution=True,observe_service_timing=True,
        observe_transport=True,arrival_rate=None,arrival_seed=None,arrival_offsets=[0.]*8,
        phase='DISJOINT_SERIAL_CUDA_EVENT_CALIBRATION_NOT_ONLINE_SAFETY',
        native_partial_cut=cfg['depth']-loops,D_loop_quantum=loops)
    source=DEEP/'data'/f'calibration_{model}_L{cfg["length"]}.json'
    pool=json.loads(source.read_text())[:8]
    main={v['prompt_id'] for v in json.loads((ROOT/'data'/cfg['dataset']).read_text())}
    assert not main&{v['prompt_id'] for v in pool}
    dataset=f'primitive_{model}_disjoint8.json'
    if not (ROOT/'data'/dataset).exists():write(ROOT/'data'/dataset,pool)
    cfg.update(dataset=dataset,prompt_ids=[v['prompt_id'] for v in pool],request_caps=[cfg['cap']]*8)
    cfg['data_scope']=dict(distinct_prompt_ids=8,disjoint_from_primary_pool=True,
        scope='Calibration only. CUDA Event enclosures are not clean kernel union.')
    return cfg


def main():
    prepare();jobs=[]
    for model in ('ouro','huginn'):
        for dynamic in (False,True):
            cfg=token_cell(model,dynamic);save(cfg);jobs.append(cfg['name'])
        for width in (16,32,64,128,256,4096 if model=='ouro' else 2048):
            cfg=primitive(model,width);save(cfg);jobs.append(cfg['name'])
    queue=json.loads((ROOT/'state/queue.json').read_text())
    write(ROOT/'state/queue.json',queue+jobs)


if __name__=='__main__':main()
