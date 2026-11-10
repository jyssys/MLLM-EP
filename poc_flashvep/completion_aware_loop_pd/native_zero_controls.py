"""Separate Native-path provisioning and wire-format cost from Loop negotiation."""
import argparse
import copy
from .common import *
from .secondary import recovery_config


def zero_config(model,name,count,seed):
    cfg=recovery_config(model,name,'native',count,seed)
    cfg.update(native_path=True,feature_native_wire_v5=True,native_partial_cut=None,native_frontier=None,
        control_arm='Native-path zero-offload',phase='NATIVE_PATH_PROVISIONING_CONTROL',
        comparison_reference=f'recovery_{model}_native_seed{seed}')
    return cfg


def main():
    p=argparse.ArgumentParser();p.add_argument('stage',choices=('N8','N32','confirmation'));a=p.parse_args()
    initialize();jobs=[]
    if a.stage in ('N8','N32'):
        count=8 if a.stage=='N8' else 32
        for model in ('ouro','huginn'):
            cfg=zero_config(model,f'control_{model}_zero_state{count}',count,2026110920)
            reference=json.loads((ROOT/'configs'/f'secondary_{model}_loop_state{count}.json').read_text())
            for key in ('state_probe','verify_transfer','async_prompt_state_probe','async_verify_handoff',
                    'state_probe_measured_only','state_hash_defer_until_close','qualification_ignore_observer_tail_latch',
                    'state_pinned_slots','arrival_offsets','arrival_seed','arrival_rate','timeout','native_transport_limits'):
                cfg[key]=copy.deepcopy(reference[key])
            cfg['require_actual_offload_qualification']=False
            if count==32:cfg['required_qualification']=f'control_{model}_zero_state8'
            save(cfg);jobs.append(cfg['name'])
    else:
        for model in ('ouro','huginn'):
            for seed in (2026110931,2026110932,2026110933):
                name=f'control_{model}_zero_seed{seed}'
                cfg=zero_config(model,name,128,seed)
                cfg.update(required_qualification=f'control_{model}_zero_state32',
                    primary_safety_arm=f'control_{model}_zero',clean_confirmation=True,
                    phase='LATER_MATCHED_ARRIVAL_NATIVE_PATH_ZERO_CONTROL')
                save(cfg);jobs.append(name)
    write(ROOT/'state'/f'zero_{a.stage}_jobs.json',jobs);print(' '.join(jobs))


if __name__=='__main__':
    main()
