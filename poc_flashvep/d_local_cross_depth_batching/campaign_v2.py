"""Qualify the role-scoped P-atomic/D-chunked contract. Do not relax state tests."""
import time
from .prepare import *
from .run import execute
from .qualification import compare


def dispatch(cfg,stage):
    save(cfg)
    write(ROOT/'state/controller_v2.json',dict(stage=stage,name=cfg['name'],time=time.time()))
    return execute(cfg['name'])


def main():
    initialize()
    valid={}
    for model in ('ouro','huginn'):
        reference=f'local_{model}_native_n8_s2026110701_off_state'
        full=f'local_{model}_full_n8_s2026110701_off_state'+('_v4' if model=='ouro' else '')
        valid[model,'full']=compare(reference,full,8)['passed']
        for method in ('chunked','paced'):
            cfg=configuration(model,method,8,verify=True)
            cfg.update(name=f'qual8_{model}_{method}_scoped_v2',prime_all_graph_shapes=False,
                deep_serial_primary_warmup=False)
            valid[model,method]=dispatch(cfg,'ROLE_SCOPED_N8') and compare(reference,cfg['name'],8)['passed']
    for model in ('ouro','huginn'):
        reference=configuration(model,'native',32,verify=True)
        reference.update(name=f'qual32_{model}_native_v2',arrival_offsets=[0.]*32,
            prime_all_graph_shapes=False,deep_serial_primary_warmup=False)
        if not dispatch(reference,'N32_REFERENCE'):
            continue
        for method in ('full','chunked','paced'):
            if not valid.get((model,method)):
                continue
            cfg=configuration(model,method,32,verify=True)
            cfg.update(name=f'qual32_{model}_{method}_scoped_v2',arrival_offsets=[0.]*32,
                prime_all_graph_shapes=False,deep_serial_primary_warmup=False)
            if dispatch(cfg,'N32_LOCAL'):
                compare(reference['name'],cfg['name'],32)
    write(ROOT/'state/controller_v2.json',dict(stage='QUALIFICATION_COMPLETE',time=time.time()))


if __name__=='__main__':main()
