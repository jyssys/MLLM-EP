"""Measure serial and burst costs on data that never enters confirmation."""
from .common import *
from .prepare import configuration, permitted_methods
from .serving import evaluate


def main(gate, serving_ledger):
    initialize()
    ledger = dict(status='CALIBRATION_IN_PROGRESS', results=[])
    target = ROOT / 'state/cost_calibration.json'
    for model in ('huginn', 'ouro'):
        qualified = gate['models'][model]['qualification']
        if not qualified.get('native', {}).get('passed'):
            ledger.setdefault('blocked_models', []).append(model)
            write(target, ledger)
            continue
        resource_blocked = []
        for result in serving_ledger['models'][model]['discovery']:
            if result.get('complete'):
                continue
            log = ROOT / 'logs' / (result['name'] + '.log')
            if log.exists() and 'torch.OutOfMemoryError:' in log.read_text(errors='replace'):
                resource_blocked.append(result['method'])
        if resource_blocked:
            ledger.setdefault('blocked_actions', {})[model] = dict(methods=sorted(set(resource_blocked)),
                status='BLOCKED_RESOURCE_OOM_AFTER_STATE_QUALIFICATION',
                scope='Retain failed discovery evidence. Do not expand the same OOM path into calibration or an oracle.')
            write(target, ledger)
        methods = [method for method in permitted_methods(model) if qualified.get(method, {}).get('passed')
                   and method not in resource_blocked]
        if 'native' not in methods:
            ledger.setdefault('blocked_models', []).append(model)
            write(target, ledger)
            continue
        methods = ['native'] + [method for method in methods if method != 'native']
        pool_path = ROOT / 'data' / f'calibration_{model}_L{2048 if model == "huginn" else 4096}.json'
        pool = json.loads(pool_path.read_text())
        batch_name = pool_path.name.replace('calibration_', 'calibration_batch_')
        batch_file = ROOT / 'data' / batch_name
        if batch_file.exists():
            assert json.loads(batch_file.read_text()) == pool[8:]
        else:
            write(batch_file, pool[8:])
        for mode, count in (('serial', 8), ('burst', 24)):
            reference = None
            for method in methods:
                if time.time() >= LONG_START_CUTOFF - 1800:
                    ledger['status'] = 'NOT_RUN_TIME_BUDGET'
                    write(target, ledger)
                    return
                cfg = configuration(model, method, count, f'f_{model}_{method}_{mode}_cost{count}',
                                    split='calibration', markers=True)
                cfg.update(oracle_calibration=True, observe_device_execution=True,
                    deep_retire_geometry_hook_after_warmup=True,
                    calibration_serial=mode == 'serial', phase='INSTRUMENTED_CALIBRATION_NOT_CLEAN_SERVING',
                    seed=202611055001, arrival_rate=None, arrival_offsets=[index * .00001 for index in range(count)],
                    arrival_pattern='SERIAL_CALIBRATION' if mode == 'serial' else 'CONTROLLED_BOUNDED_BURST')
                if mode == 'burst':
                    cfg.update(dataset=batch_name, prompt_ids=[row['prompt_id'] for row in pool[8:]])
                result = evaluate(cfg, reference)
                ledger['results'].append(result)
                write(target, ledger)
                if method == 'native':
                    if not result.get('passed'):
                        break
                    reference = cfg['name']
    ledger['status'] = 'CALIBRATION_FINISHED'
    write(target, ledger)
