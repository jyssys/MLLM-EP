"""Bounded phase controller. Expand only paths with complete exact evidence."""
import time
from .prepare import *
from .run import execute
from .qualification import compare


def main():
    prepare()
    pending = [
        'local_ouro_native_n8_s2026110701_off_state',
        'local_ouro_full_n8_s2026110701_off_state_v4',
        'local_huginn_native_n8_s2026110701_off_state',
        'local_huginn_full_n8_s2026110701_off_state',
        'local_ouro_chunked_n8_s2026110701_off_state',
        'micro_huginn_c32_b2_same_v2', 'activity_collector_selftest']
    while any(not (ROOT/'raw'/name/'dispatcher_result.json').exists() for name in pending):
        assert time.time()<LONG_START_CUTOFF, 'Pending jobs reached the cutoff'
        write(ROOT/'state/controller.json',dict(stage='WAIT_REGISTERED_JOBS',time=time.time(),pending=[name for name in pending
            if not (ROOT/'raw'/name/'dispatcher_result.json').exists()]))
        time.sleep(5)
    for model in ('ouro','huginn'):
        for method in ('chunked','paced'):
            cfg = configuration(model,method,8,verify=True)
            cfg.update(name=cfg['name']+'_limited_warm',prime_all_graph_shapes=False,deep_serial_primary_warmup=False)
            save(cfg)
            if not execute(cfg['name']):
                continue
            reference = f'local_{model}_native_n8_s2026110701_off_state'
            result = compare(reference,cfg['name'],8)
            write(ROOT/'state/controller.json',dict(stage='N8_GATE',model=model,method=method,passed=result['passed'],time=time.time()))
    for model in ('ouro','huginn'):
        reference = f'local_{model}_native_n8_s2026110701_off_state'
        full = f'local_{model}_full_n8_s2026110701_off_state'+('_v4' if model=='ouro' else '')
        if (ROOT/'raw'/full/'requests.jsonl').exists():
            compare(reference,full,8)
        reference32 = configuration(model,'native',32,verify=True)
        reference32.update(name=f'qual32_{model}_native',arrival_offsets=[0.]*32,
            prime_all_graph_shapes=False,deep_serial_primary_warmup=False)
        save(reference32)
        if not execute(reference32['name']):
            continue
        for method in ('full','chunked','paced'):
            n8 = full if method=='full' else f'local_{model}_{method}_n8_s2026110701_off_state_limited_warm'
            path = ROOT/'raw'/n8/'local_qualification.json'
            if not path.exists() or not json.loads(path.read_text())['passed']:
                continue
            cfg = configuration(model,method,32,verify=True)
            cfg.update(name=f'qual32_{model}_{method}',arrival_offsets=[0.]*32,
                prime_all_graph_shapes=False,deep_serial_primary_warmup=False)
            save(cfg)
            if execute(cfg['name']):
                compare(reference32['name'],cfg['name'],32)
    write(ROOT/'state/controller.json',dict(stage='QUALIFICATION_COMPLETE',time=time.time()))


if __name__ == '__main__':
    main()
