"""Measure real full-prompt suffixes without calling serial timing serving slack."""
import copy
import shutil
import time
from .prepare import *
from .run import execute


def primitive(model,method,cut=None):
    source_method='native' if method=='native' else ('loop3' if model=='ouro' else 'loop31')
    source=DEEP/'configs'/f'b_{model}_{source_method}_reserved8_state8.json'
    cfg=copy.deepcopy(json.loads(source.read_text()))
    length=4096 if model=='ouro' else 2048
    path=DEEP/'data'/f'calibration_{model}_L{length}.json'
    pool=json.loads(path.read_text())[:8];target=ROOT/'data'/f'primitive_{model}_disjoint8.json'
    if not target.exists():write(target,pool)
    main_ids={v['prompt_id'] for v in json.loads((ROOT/'data'/f'heldout_{model}_L{length}.json').read_text())}
    assert not main_ids&{v['prompt_id'] for v in pool},'Calibration texts overlap the primary pool'
    cfg.update(name=f'p_{model}_{method}_serial8_v1',method=method,count=8,
        state_probe=False,verify_transfer=False,calibration_serial=True,observe_device_execution=True,
        observe_service_timing=True,observe_transport=True,idle_campaign=True,idle_observer='off',
        dlocal_campaign=False,D_local_probe_output_pacing=False,deep_retire_geometry_hook_after_warmup=True,
        deep_complete_D_geometry_warmup=False,
        phase='INSTRUMENTED_SERIAL_SUFFIX_CALIBRATION_NOT_CLEAN_SERVING',
        seed=202611061001,arrival_seed=None,arrival_rate=None,arrival_offsets=[0.]*8,
        dataset=target.name,prompt_ids=[v['prompt_id'] for v in pool],request_caps=[cfg['cap']]*8,
        timeout=600,startup_timeout=300,load='serial',
        native_path=method!='native',feature_native_wire_v5=method!='native',
        parent_configuration=dict(path=str(source),SHA256=digest(source),scope='Previously qualified same-kernel path. New counts require their own output/work checks.'))
    cfg.pop('native_request_cuts',None);cfg.pop('native_frontier',None)
    if cut is None:cfg.pop('native_partial_cut',None)
    else:cfg['native_partial_cut']=cut
    return cfg


def main():
    prepare()
    while time.time()<LONG_START_CUTOFF:
        path=ROOT/'state/context_trials.json'
        if path.exists() and json.loads(path.read_text())['status']=='COMPLETE':break
        time.sleep(5)
    results=[]
    for model in ('ouro','huginn'):
        depth=4 if model=='ouro' else 32
        for loops in (0,1,2,3) if model=='ouro' else (0,1,2,4,8):
            method='native' if loops==0 else f'loop_suffix{loops}'
            cfg=primitive(model,method,None if loops==0 else depth-loops);save(cfg)
            write(ROOT/'state/primitive_full.json',dict(time=time.time(),status='RUNNING',name=cfg['name'],results=results))
            complete=execute(cfg['name']);results.append(dict(model=model,name=cfg['name'],complete=complete,remaining_loops=loops))
    write(ROOT/'state/primitive_full.json',dict(time=time.time(),status='COMPLETE',results=results))


if __name__=='__main__':main()
