"""Diagnose first-shape timing drift and measure a warmed B1 generation control."""
import json
import subprocess
import time
from .common import *

def generation(a,ep,ids,cap):
    import torch
    import torch.distributed as dist
    prefix=ids.clone();tokens=[];stamps=[]
    dist.barrier();torch.cuda.synchronize()
    start=time.perf_counter()
    for _ in range(cap):
        ep.reset()
        logits=a.forward(prefix)
        token=logits[:,-1].argmax(-1,keepdim=True)
        torch.cuda.synchronize()
        tokens.append(token.item());stamps.append(time.perf_counter()-start)
        prefix=torch.cat([prefix,token],dim=-1)
    return dict(ids=tokens,timestamps_s=stamps,e2e_s=stamps[-1],ttft_s=stamps[0],
        tpot_s=(stamps[-1]-stamps[0])/(cap-1),cap=cap,eos_enabled=False)

def measure(a,ep,target,config):
    import torch
    from .measure import shaped_inputs
    ids,_=shaped_inputs(a,1,128)
    for _ in range(10):
        ep.reset();a.forward(ids)
    control=[]
    for trial in range(4):
        hardware=subprocess.check_output(['nvidia-smi','-i',str(config['gpus'][ep.rank]),
            '--query-gpu=index,uuid,temperature.gpu,clocks.sm,power.draw,memory.used','--format=csv'],text=True)
        value=generation(a,ep,ids,128)
        value.update(trial=trial,kind='FIRST_FULL_GENERATION_SHAPE' if trial==0 else 'WARMED_FULL_GENERATION',
            model=a.name,ep=ep.world,rank=ep.rank,hardware_before=hardware,
            evidence='LIVE_UNINSTRUMENTED_GENERATION_SEQUENCE',warmup_full_generations=trial)
        if control:assert value['ids']==control[0]['ids'],'Warmup changes generated tokens'
        control.append(value)
        append(target/f'steady_generation.rank{ep.rank}.jsonl',value)
    short=generation(a,ep,ids,32)
    append(target/f'steady_generation.rank{ep.rank}.jsonl',dict(short,trial=4,kind='WARMED_32_TOKEN_GENERATION',
        model=a.name,ep=ep.world,rank=ep.rank,evidence='LIVE_UNINSTRUMENTED_GENERATION_SEQUENCE',warmup_full_generations=4))
    ratio=control[0]['e2e_s']/control[-1]['e2e_s']
    stability=control[-1]['e2e_s']/control[-2]['e2e_s']
    write(target/f'shape_drift.rank{ep.rank}.json',dict(first_to_last_ratio=ratio,
        last_two_ratio=stability,last_two_within_10pct=.9<=stability<=1.1,
        token_exact_all=True,scope='Same uninstrumented path. First shape preparation versus warmed repetition.'))

def main():
    from .launch import launch
    while not (ROOT/'campaign_ready_probe.json').exists() or len(json.loads((ROOT/'campaign_ready_probe.json').read_text()))<6:
        time.sleep(5)
    results=[]
    for world in [4,8]:
        for model in ['foil1','ifm_ouro','ifm_huginn']:
            config=dict(id=f'{model}_ep{world}_steady_generation_v1',model=model,gpus=list(range(world)),
                phase='steady_generation',timeout_s=3600)
            try:results.append(launch(config))
            except Exception as error:results.append(dict(id=config['id'],status='FAILED',error=repr(error)))
            write(ROOT/'campaign_steady_generation.json',results)

if __name__=='__main__':main()
